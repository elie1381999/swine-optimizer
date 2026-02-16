from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from scipy.optimize import linprog
from typing import List, Dict, Optional, Any

app = FastAPI(title="Swine Feed Optimizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OptimizeRequest(BaseModel):
    phase_name: str
    animal_type: str
    requirements: Dict[str, Any]
    ingredients: List[Dict[str, Any]]
    dmi_adjusted_kg: float = 2.5

@app.get("/")
def root():
    return {"service": "Swine Feed Optimizer API", "status": "online", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/optimize")
def optimize_feed(request: OptimizeRequest):
    try:
        ingredients = request.ingredients
        requirements = request.requirements
        phase_name = request.phase_name
        dmi_adjusted_kg = request.dmi_adjusted_kg
        
        if not ingredients or len(ingredients) < 5:
            raise HTTPException(status_code=400, detail="Minimum 5 ingredients required")
        
        def check_price_mode(ingredients):
            prices = [float(ing.get('price_per_kg', 0)) for ing in ingredients]
            valid_prices = [p for p in prices if p > 0]
            if len(valid_prices) == 0:
                return {"mode": "single_formula", "reason": "no_prices"}
            if len(set(valid_prices)) == 1:
                return {"mode": "single_formula", "reason": "identical_prices"}
            return {"mode": "three_formulas", "reason": "varied_prices"}
        
        price_check = check_price_mode(ingredients)
        n = len(ingredients)
        c_cost = [float(ing.get('price_per_kg', 0.50)) for ing in ingredients]
        
        if price_check['mode'] == 'single_formula':
            avg_price = sum(c_cost) / len([p for p in c_cost if p > 0]) if any(p > 0 for p in c_cost) else 0.50
            c_cost = [p if p > 0 else avg_price for p in c_cost]
        
        A_ub = []
        b_ub = []
        constraint_names = []
        
        def add_min(key, target, name):
            if target is None or target == 0:
                return
            row = [-float(ing.get(key, 0) or 0) for ing in ingredients]
            if all(val == 0 for val in row):
                return
            A_ub.append(row)
            b_ub.append(-float(target))
            constraint_names.append(f"MIN_{name}")
        
        def add_max(key, limit, name):
            if limit is None or limit == 0:
                return
            row = [float(ing.get(key, 0) or 0) for ing in ingredients]
            if all(val == 0 for val in row):
                return
            A_ub.append(row)
            b_ub.append(float(limit))
            constraint_names.append(f"MAX_{name}")
        
        add_min('ne_swine_kcal_kg', requirements.get('ne_min'), 'ENERGY')
        add_min('sid_lysine_pct', requirements.get('sid_lysine_min'), 'LYSINE')
        add_min('sttd_phosphorus_pct', requirements.get('sttd_phosphorus_min'), 'PHOSPHORUS')
        add_max('crude_fiber_pct', requirements.get('max_fiber'), 'FIBER')
        add_max('adf_pct', requirements.get('max_adf'), 'ADF')
        
        core_nutrient_count = len(A_ub)
        
        ratios = requirements.get('ideal_protein_ratios', {})
        ratio_map = {
            'sid_threonine_pct': 'threonine',
            'sid_methionine_cysteine_pct': 'methionine_cysteine',
            'sid_tryptophan_pct': 'tryptophan',
            'sid_valine_pct': 'valine',
            'sid_isoleucine_pct': 'isoleucine'
        }
        
        for db_col, ratio_name in ratio_map.items():
            if ratio_name in ratios:
                target_ratio = float(ratios[ratio_name])
                row = []
                for ing in ingredients:
                    lys = float(ing.get('sid_lysine_pct', 0) or 0)
                    aa = float(ing.get(db_col, 0) or 0)
                    row.append((target_ratio * lys) - aa)
                A_ub.append(row)
                b_ub.append(0)
                constraint_names.append(f"RATIO_{ratio_name.upper()}")
        
        row_ca_max = []
        row_ca_min = []
        has_minerals = False
        
        for ing in ingredients:
            ca = float(ing.get('ca_percentage', 0) or 0)
            p = float(ing.get('sttd_phosphorus_pct', 0) or 0)
            if ca > 0 or p > 0:
                has_minerals = True
            row_ca_max.append(ca - (1.35 * p))
            row_ca_min.append(-1 * (ca - (1.10 * p)))
        
        if has_minerals:
            A_ub.append(row_ca_max)
            b_ub.append(0)
            constraint_names.append("MAX_CA_P_RATIO")
            A_ub.append(row_ca_min)
            b_ub.append(0)
            constraint_names.append("MIN_CA_P_RATIO")
        
        bounds = []
        phase_restricted = []
        
        for ing in ingredients:
            lower = 0.0
            upper = 1.0
            suitable_phases = ing.get('suitable_for_pig_phases', [])
            if suitable_phases and phase_name not in suitable_phases:
                upper = 0.0
                phase_restricted.append(ing['name'])
            limits = ing.get('max_inclusion_pct', {})
            if isinstance(limits, dict) and phase_name in limits:
                limit_decimal = float(limits[phase_name]) / 100.0
                if limit_decimal < upper:
                    upper = limit_decimal
            bounds.append((lower, upper))
        
        A_eq = [[1] * n]
        b_eq = [1]
        
        solutions = []
        failed_strategies = []
        
        def run_optimization(strategy_name, label, description, safety_margin, b_ub_base):
            try:
                b_ub_modified = b_ub_base.copy()
                for i in range(min(3, core_nutrient_count)):
                    if b_ub_modified[i] < 0:
                        b_ub_modified[i] = b_ub_modified[i] * safety_margin
                result = linprog(c_cost, A_ub=A_ub, b_ub=b_ub_modified, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
                if result.success:
                    return {"strategy": strategy_name, "label": label, "description": description, "safety_margin": f"+{int((safety_margin - 1) * 100)}%" if safety_margin > 1 else "0%", "result": result}
                else:
                    failed_strategies.append({"strategy": strategy_name, "reason": result.message})
                    return None
            except Exception as e:
                failed_strategies.append({"strategy": strategy_name, "reason": str(e)})
                return None
        
        if price_check['mode'] == 'single_formula':
            sol = run_optimization("balanced", "⚖️ Optimal Formula", "Balanced nutritional profile", 1.05, b_ub)
            if sol:
                solutions.append(sol)
        else:
            sol1 = run_optimization("cheapest", "💰 Lowest Cost", "Minimum cost formula", 1.00, b_ub)
            if sol1:
                solutions.append(sol1)
            sol2 = run_optimization("performance", "🏆 Best Performance", "High performance formula", 1.12, b_ub)
            if sol2:
                solutions.append(sol2)
            sol3 = run_optimization("balanced", "⚖️ Balanced", "Balanced formula", 1.05, b_ub)
            if sol3:
                solutions.append(sol3)
        
        if not solutions:
            return {"status": "infeasible", "message": "Cannot create valid formula", "diagnostics": ["Check ingredients"]}
        
        formatted_solutions = []
        for sol in solutions:
            result = sol['result']
            mix = []
            total_cost = 0
            analysis = {"ne_swine_kcal": 0, "sid_lysine_pct": 0, "crude_fiber_pct": 0, "calcium_pct": 0, "sttd_phosphorus_pct": 0, "crude_protein": 0, "adf_pct": 0, "ndf_pct": 0}
            
            for i, pct in enumerate(result.x):
                if pct > 0.001:
                    ing = ingredients[i]
                    cost_contrib = pct * c_cost[i]
                    total_cost += cost_contrib
                    for key in analysis:
                        analysis[key] += pct * float(ing.get(key, 0) or 0)
                    mix.append({"id": ing.get('id'), "name": ing.get('name'), "percentage": round(pct * 100, 2), "kg_per_ton": round(pct * 1000, 1), "kg_per_100kg": round(pct * 100, 1), "kg_per_pig_daily": round(pct * dmi_adjusted_kg, 3), "cost_per_kg": ing.get('price_per_kg', 0), "cost_contribution_per_ton": round(cost_contrib * 1000, 2), "is_synthetic": ing.get('is_synthetic_aa', False), "price_source": ing.get('price_source', 'unknown')})
            
            mix.sort(key=lambda x: x['percentage'], reverse=True)
            ca_to_p = (analysis['calcium_pct'] / analysis['sttd_phosphorus_pct'] if analysis['sttd_phosphorus_pct'] > 0 else 0)
            
            formatted_solutions.append({"strategy": sol['strategy'], "label": sol['label'], "description": sol['description'], "safety_margin": sol['safety_margin'], "mix": mix, "cost_per_ton": round(total_cost * 1000, 2), "cost_per_kg": round(total_cost, 4), "cost_per_pig_daily": round(total_cost * dmi_adjusted_kg, 3), "cost_per_pig_monthly": round(total_cost * dmi_adjusted_kg * 30, 2), "daily_feed_per_pig_kg": round(dmi_adjusted_kg, 2), "monthly_feed_per_pig_kg": round(dmi_adjusted_kg * 30, 1), "analysis": {k: round(v, 2) for k, v in analysis.items()}, "ca_to_p_ratio": round(ca_to_p, 2), "ratio_status": "✅ Healthy" if 1.1 <= ca_to_p <= 1.35 else "⚠️ Check ratio", "ingredient_count": len(mix), "iterations": result.nit})
        
        return {"status": "success", "optimization_mode": price_check['mode'], "mode_reason": price_check['reason'], "solutions": formatted_solutions, "requirements_used": requirements, "phase": phase_name, "total_ingredients_available": len(ingredients), "phase_restricted_count": len(phase_restricted)}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
