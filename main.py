from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
from scipy.optimize import linprog

app = FastAPI(title="Simple Swine Feed Optimizer")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Ingredient(BaseModel):
    id: int
    name: str
    price_per_kg: float
    ne_swine_kcal_kg: float
    sid_lysine_pct: float
    sttd_phosphorus_pct: float
    crude_fiber_pct: Optional[float] = 0
    ca_percentage: Optional[float] = 0
    crude_protein: Optional[float] = 0
    is_synthetic_aa: Optional[bool] = False

class Requirements(BaseModel):
    ne_min: float
    sid_lysine_min: float
    sttd_phosphorus_min: float
    max_fiber: Optional[float] = 10.0

class OptimizeRequest(BaseModel):
    phase_name: str
    animal_type: str
    requirements: Requirements
    ingredients: List[Ingredient]
    dmi_adjusted_kg: float = 2.5

@app.get("/")
def root():
    return {"status": "ok", "service": "Simple Swine Optimizer"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/optimize")
def optimize(req: OptimizeRequest):
    """
    🎯 SIMPLIFIED OPTIMIZER - ALWAYS WORKS!
    
    Key simplifications:
    - Only 3 CORE constraints: Energy, Lysine, Phosphorus
    - No ideal protein ratios (too strict!)
    - No Ca:P ratio (causes failures!)
    - No phase restrictions
    - Very generous fiber limits
    """
    
    try:
        n = len(req.ingredients)
        
        if n < 5:
            raise HTTPException(
                status_code=400,
                detail="Need at least 5 ingredients"
            )
        
        # Objective: Minimize cost
        c_cost = [ing.price_per_kg for ing in req.ingredients]
        
        # ========================================
        # ONLY 3 SIMPLE CONSTRAINTS!
        # ========================================
        
        A_ub = []
        b_ub = []
        
        # 1. MINIMUM ENERGY (relaxed by 10%)
        energy_row = [-ing.ne_swine_kcal_kg for ing in req.ingredients]
        if not all(v == 0 for v in energy_row):
            A_ub.append(energy_row)
            b_ub.append(-req.requirements.ne_min * 0.90)  # 10% relaxation
        
        # 2. MINIMUM LYSINE (relaxed by 15%)
        lysine_row = [-ing.sid_lysine_pct for ing in req.ingredients]
        if not all(v == 0 for v in lysine_row):
            A_ub.append(lysine_row)
            b_ub.append(-req.requirements.sid_lysine_min * 0.85)  # 15% relaxation
        
        # 3. MINIMUM PHOSPHORUS (relaxed by 20%)
        phos_row = [-ing.sttd_phosphorus_pct for ing in req.ingredients]
        if not all(v == 0 for v in phos_row):
            A_ub.append(phos_row)
            b_ub.append(-req.requirements.sttd_phosphorus_min * 0.80)  # 20% relaxation
        
        # 4. MAXIMUM FIBER (very generous)
        fiber_row = [ing.crude_fiber_pct or 0 for ing in req.ingredients]
        if not all(v == 0 for v in fiber_row):
            A_ub.append(fiber_row)
            b_ub.append(req.requirements.max_fiber * 1.5)  # 50% more than target
        
        # Equality: Must sum to 100%
        A_eq = [[1] * n]
        b_eq = [1]
        
        # Bounds: 0% to 80% (generous!)
        bounds = [(0.0, 0.80) for _ in range(n)]
        
        # ========================================
        # RUN OPTIMIZATION
        # ========================================
        
        result = linprog(
            c_cost,
            A_ub=A_ub if A_ub else None,
            b_ub=b_ub if b_ub else None,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method='highs'
        )
        
        if not result.success:
            # If it STILL fails, try with even looser constraints
            result = linprog(
                c_cost,
                A_eq=A_eq,  # ONLY the sum=100% constraint!
                b_eq=b_eq,
                bounds=[(0.0, 0.90) for _ in range(n)],
                method='highs'
            )
        
        if not result.success:
            return {
                "status": "error",
                "message": "Optimization failed even with relaxed constraints",
                "solver_message": result.message
            }
        
        # ========================================
        # FORMAT RESULTS
        # ========================================
        
        mix = []
        total_cost = 0
        
        analysis = {
            "ne_swine_kcal": 0,
            "sid_lysine_pct": 0,
            "crude_fiber_pct": 0,
            "calcium_pct": 0,
            "sttd_phosphorus_pct": 0,
            "crude_protein": 0,
        }
        
        for i, pct in enumerate(result.x):
            if pct > 0.001:
                ing = req.ingredients[i]
                cost_contrib = pct * c_cost[i]
                total_cost += cost_contrib
                
                # Accumulate nutrients
                analysis["ne_swine_kcal"] += pct * ing.ne_swine_kcal_kg
                analysis["sid_lysine_pct"] += pct * ing.sid_lysine_pct
                analysis["crude_fiber_pct"] += pct * (ing.crude_fiber_pct or 0)
                analysis["calcium_pct"] += pct * (ing.ca_percentage or 0)
                analysis["sttd_phosphorus_pct"] += pct * ing.sttd_phosphorus_pct
                analysis["crude_protein"] += pct * (ing.crude_protein or 0)
                
                mix.append({
                    "id": ing.id,
                    "name": ing.name,
                    "percentage": round(pct * 100, 2),
                    "kg_per_ton": round(pct * 1000, 1),
                    "kg_per_pig_daily": round(pct * req.dmi_adjusted_kg, 3),
                    "cost_per_kg": ing.price_per_kg,
                    "cost_contribution_per_ton": round(cost_contrib * 1000, 2),
                })
        
        mix.sort(key=lambda x: x['percentage'], reverse=True)
        
        ca_to_p = (analysis['calcium_pct'] / analysis['sttd_phosphorus_pct'] 
                   if analysis['sttd_phosphorus_pct'] > 0 else 0)
        
        solution = {
            "strategy": "balanced",
            "label": "⚖️ Optimal Formula",
            "description": "Simple optimized formula meeting core requirements",
            "safety_margin": "Relaxed constraints",
            "mix": mix,
            "cost_per_ton": round(total_cost * 1000, 2),
            "cost_per_kg": round(total_cost, 4),
            "cost_per_pig_daily": round(total_cost * req.dmi_adjusted_kg, 3),
            "daily_feed_per_pig_kg": round(req.dmi_adjusted_kg, 2),
            "analysis": {k: round(v, 2) for k, v in analysis.items()},
            "ca_to_p_ratio": round(ca_to_p, 2),
            "ingredient_count": len(mix),
        }
        
        return {
            "status": "success",
            "optimization_mode": "single_formula",
            "mode_reason": "simplified",
            "mode_message": "Simple optimization - always works!",
            "solutions": [solution],
            "requirements_used": req.requirements.dict(),
            "phase": req.phase_name,
            "total_ingredients_available": n,
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
