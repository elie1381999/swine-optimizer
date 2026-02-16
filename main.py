from functions_framework import http
import json
import numpy as np
from scipy.optimize import linprog

@http
def swine_mix_optimizer(request):
    """
    🎯 PRODUCTION-GRADE SWINE FEED OPTIMIZER
    
    Creates 1-3 optimized feed mixes from user's available ingredients.
    
    MODES:
    - No prices → 1 formula (balanced, focus on ratios/quantities)
    - Same prices → 1 formula (can't optimize cost)
    - Varied prices → 3 formulas (cheapest, performance, balanced)
    
    KEY FEATURES:
    - NRC 2012 compliant
    - Ideal protein ratios enforced
    - Ca:P balance (1.1-1.35:1)
    - Phase-specific ingredient limits
    - Smart diagnostics on failure
    """
    
    # ============================================================
    # CORS HANDLING
    # ============================================================
    if request.method == 'OPTIONS':
        return '', 204, {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        }
    
    headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
    }
    
    # ============================================================
    # INPUT VALIDATION
    # ============================================================
    try:
        data = request.get_json()
        ingredients = data['ingredients']
        requirements = data['requirements']
        phase_name = data['phase_name']
        dmi_adjusted_kg = data.get('dmi_adjusted_kg', 2.5)  # Daily feed intake
        
        if not ingredients or len(ingredients) == 0:
            return json.dumps({
                "status": "error",
                "message": "No ingredients provided"
            }), 400, headers
        
        if len(ingredients) < 5:
            return json.dumps({
                "status": "error",
                "message": f"Only {len(ingredients)} ingredients provided. Minimum 5 required.",
                "hint": "Need: Energy source (Corn), Protein (SBM), Minerals (Limestone, DCP), Salt"
            }), 400, headers
        
    except KeyError as e:
        return json.dumps({
            "status": "error",
            "message": f"Missing required field: {str(e)}"
        }), 400, headers
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Invalid input: {str(e)}"
        }), 400, headers
    
    # ============================================================
    # PRICE AVAILABILITY CHECK
    # ============================================================
    def check_price_mode(ingredients):
        """Determine optimization mode based on price data"""
        prices = [float(ing.get('price_per_kg', 0)) for ing in ingredients]
        price_sources = [ing.get('price_source', 'unknown') for ing in ingredients]
        
        # Count non-zero prices
        valid_prices = [p for p in prices if p > 0]
        
        if len(valid_prices) == 0:
            return {
                "mode": "single_formula",
                "reason": "no_prices",
                "message": "No price data - creating single formula based on nutritional balance"
            }
        
        # Check if all prices are identical
        unique_prices = len(set(valid_prices))
        if unique_prices == 1:
            return {
                "mode": "single_formula",
                "reason": "identical_prices",
                "message": "All ingredients have same price - creating single formula"
            }
        
        # Check price quality
        user_count = price_sources.count('user_local')
        estimated_count = price_sources.count('estimated')
        
        return {
            "mode": "three_formulas",
            "reason": "varied_prices",
            "message": "Multiple price points available - creating 3 optimized solutions",
            "price_quality": {
                "user_prices": user_count,
                "estimated_prices": estimated_count,
                "total": len(ingredients)
            }
        }
    
    price_check = check_price_mode(ingredients)
    
    # ============================================================
    # PREPARE BASE CONSTRAINTS (SAME FOR ALL SOLUTIONS)
    # ============================================================
    
    n = len(ingredients)
    
    # Objective: Minimize COST (always)
    c_cost = [float(ing.get('price_per_kg', 0.50)) for ing in ingredients]
    
    # Replace zeros with average price (for single-formula mode)
    if price_check['mode'] == 'single_formula':
        avg_price = sum(c_cost) / len([p for p in c_cost if p > 0]) if any(p > 0 for p in c_cost) else 0.50
        c_cost = [p if p > 0 else avg_price for p in c_cost]
    
    # INEQUALITY CONSTRAINTS (A_ub @ x <= b_ub)
    A_ub = []
    b_ub = []
    constraint_names = []
    
    def add_min(key, target, name):
        """Add minimum constraint (>=)"""
        if target is None or target == 0:
            return
        row = [-float(ing.get(key, 0) or 0) for ing in ingredients]
        if all(val == 0 for val in row):
            return
        A_ub.append(row)
        b_ub.append(-float(target))
        constraint_names.append(f"MIN_{name}")
    
    def add_max(key, limit, name):
        """Add maximum constraint (<=)"""
        if limit is None or limit == 0:
            return
        row = [float(ing.get(key, 0) or 0) for ing in ingredients]
        if all(val == 0 for val in row):
            return
        A_ub.append(row)
        b_ub.append(float(limit))
        constraint_names.append(f"MAX_{name}")
    
    # CORE NUTRIENT CONSTRAINTS (will be scaled for performance/balanced)
    add_min('ne_swine_kcal_kg', requirements.get('ne_min'), 'ENERGY')
    add_min('sid_lysine_pct', requirements.get('sid_lysine_min'), 'LYSINE')
    add_min('sttd_phosphorus_pct', requirements.get('sttd_phosphorus_min'), 'PHOSPHORUS')
    add_max('crude_fiber_pct', requirements.get('max_fiber'), 'FIBER')
    add_max('adf_pct', requirements.get('max_adf'), 'ADF')
    
    core_nutrient_count = len(A_ub)
    
    # ============================================================
    # IDEAL PROTEIN RATIOS (THE BIOLOGY LOCK 🔐)
    # ============================================================
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
                # Logic: AA >= (Ratio × Lysine)
                # Rearranged: (Ratio × Lys) - AA <= 0
                row.append((target_ratio * lys) - aa)
            A_ub.append(row)
            b_ub.append(0)
            constraint_names.append(f"RATIO_{ratio_name.upper()}")
    
    # ============================================================
    # CALCIUM:PHOSPHORUS BALANCE (KIDNEY & BONE SAVER 🦴)
    # ============================================================
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
    
    # ============================================================
    # INGREDIENT BOUNDS (PHASE SAFETY GATEKEEPER 🚪)
    # ============================================================
    bounds = []
    phase_restricted = []
    
    for ing in ingredients:
        lower = 0.0
        upper = 1.0
        
        # Check phase suitability
        suitable_phases = ing.get('suitable_for_pig_phases', [])
        if suitable_phases and phase_name not in suitable_phases:
            upper = 0.0
            phase_restricted.append(ing['name'])
        
        # Check max inclusion from JSONB
        limits = ing.get('max_inclusion_pct', {})
        if isinstance(limits, dict) and phase_name in limits:
            limit_decimal = float(limits[phase_name]) / 100.0
            if limit_decimal < upper:
                upper = limit_decimal
        
        bounds.append((lower, upper))
    
    # EQUALITY CONSTRAINT (Must sum to 100%)
    A_eq = [[1] * n]
    b_eq = [1]
    
    # ============================================================
    # RUN OPTIMIZATION(S) BASED ON PRICE MODE
    # ============================================================
    
    solutions = []
    failed_strategies = []
    
    def run_optimization(strategy_name, label, description, safety_margin, b_ub_base):
        """Helper to run a single optimization"""
        try:
            # Apply safety margin to core nutrients
            b_ub_modified = b_ub_base.copy()
            for i in range(min(3, core_nutrient_count)):
                if b_ub_modified[i] < 0:  # It's a minimum
                    b_ub_modified[i] = b_ub_modified[i] * safety_margin
            
            result = linprog(
                c_cost,
                A_ub=A_ub,
                b_ub=b_ub_modified,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method='highs'
            )
            
            if result.success:
                return {
                    "strategy": strategy_name,
                    "label": label,
                    "description": description,
                    "safety_margin": f"+{int((safety_margin - 1) * 100)}%" if safety_margin > 1 else "0%",
                    "result": result
                }
            else:
                failed_strategies.append({
                    "strategy": strategy_name,
                    "reason": result.message
                })
                return None
        except Exception as e:
            failed_strategies.append({
                "strategy": strategy_name,
                "reason": str(e)
            })
            return None
    
    # Execute based on mode
    if price_check['mode'] == 'single_formula':
        # Just create ONE balanced formula
        sol = run_optimization(
            "balanced",
            "⚖️ Optimal Formula",
            "Balanced nutritional profile meeting all requirements",
            1.05,  # 5% safety margin
            b_ub
        )
        if sol:
            solutions.append(sol)
    else:
        # Create 3 formulas: CHEAPEST, PERFORMANCE, BALANCED
        
        # 1. CHEAPEST (exact requirements, 0% margin)
        sol1 = run_optimization(
            "cheapest",
            "💰 Lowest Cost",
            "Meets exact nutritional requirements at minimum cost",
            1.00,
            b_ub
        )
        if sol1:
            solutions.append(sol1)
        
        # 2. PERFORMANCE (12% margin)
        sol2 = run_optimization(
            "performance",
            "🏆 Best Performance",
            "Optimized for high-genetics pigs with 12% safety margin",
            1.12,
            b_ub
        )
        if sol2:
            solutions.append(sol2)
        
        # 3. BALANCED (5% margin)
        sol3 = run_optimization(
            "balanced",
            "⚖️ Balanced",
            "Practical compromise with 5% safety buffer",
            1.05,
            b_ub
        )
        if sol3:
            solutions.append(sol3)
    
    # ============================================================
    # SMART FAILURE DIAGNOSTICS
    # ============================================================
    
    if not solutions:
        diagnostics = []
        
        if len(ingredients) < 7:
            diagnostics.append(f"⚠️ Only {len(ingredients)} ingredients (recommend 7+ for flexibility)")
        
        # Check for essential ingredients
        has_energy = any(float(ing.get('ne_swine_kcal_kg', 0)) > 2000 for ing in ingredients)
        has_protein = any(float(ing.get('sid_lysine_pct', 0)) > 0.8 for ing in ingredients)
        has_minerals = any(float(ing.get('ca_percentage', 0)) > 20 for ing in ingredients)
        
        if not has_energy:
            diagnostics.append("🔴 CRITICAL: No high-energy ingredient (Corn, Wheat, Barley)")
        if not has_protein:
            diagnostics.append("🔴 CRITICAL: No protein source (Soybean Meal, Fish Meal, Synthetic Lysine)")
        if not has_minerals:
            diagnostics.append("⚠️ Missing calcium source (Limestone)")
        
        if len(phase_restricted) > len(ingredients) / 2:
            diagnostics.append(f"⚠️ Phase '{phase_name}' excludes {len(phase_restricted)} ingredients")
        
        # Check if requirements are achievable
        max_energy = sum(float(ing.get('ne_swine_kcal_kg', 0)) for ing in ingredients) / n
        if requirements.get('ne_min', 0) > max_energy * 0.8:
            diagnostics.append(
                f"🔴 Energy target ({requirements.get('ne_min')} kcal/kg) too high. "
                f"Maximum achievable: ~{int(max_energy * 0.8)} kcal/kg"
            )
        
        return json.dumps({
            "status": "infeasible",
            "message": "Cannot create valid formula with selected ingredients",
            "diagnostics": diagnostics,
            "failed_strategies": failed_strategies,
            "suggestions": [
                "✅ Add Corn or Wheat (provides energy)",
                "✅ Add Soybean Meal or Synthetic Lysine (provides protein)",
                "✅ Add Limestone and Dicalcium Phosphate (provides minerals)",
                "✅ Add Salt (essential mineral)",
                "⚠️ If all ingredients present, requirements may be too high"
            ],
            "phase_restrictions": {
                "phase": phase_name,
                "excluded_ingredients": phase_restricted
            }
        }), 200, headers
    
    # ============================================================
    # FORMAT OUTPUT WITH DETAILED ANALYSIS
    # ============================================================
    
    formatted_solutions = []
    
    for sol in solutions:
        result = sol['result']
        mix = []
        total_cost = 0
        
        # Nutritional analysis
        analysis = {
            "ne_swine_kcal": 0,
            "sid_lysine_pct": 0,
            "crude_fiber_pct": 0,
            "calcium_pct": 0,
            "sttd_phosphorus_pct": 0,
            "crude_protein": 0,
            "adf_pct": 0,
            "ndf_pct": 0
        }
        
        for i, pct in enumerate(result.x):
            if pct > 0.001:  # Filter out numerical noise
                ing = ingredients[i]
                cost_contrib = pct * c_cost[i]
                total_cost += cost_contrib
                
                # Accumulate weighted nutrients
                for key in analysis:
                    analysis[key] += pct * float(ing.get(key, 0) or 0)
                
                mix.append({
                    "id": ing.get('id'),
                    "name": ing.get('name'),
                    "percentage": round(pct * 100, 2),
                    "kg_per_ton": round(pct * 1000, 1),
                    "kg_per_100kg": round(pct * 100, 1),
                    "kg_per_pig_daily": round(pct * dmi_adjusted_kg, 3),
                    "cost_per_kg": ing.get('price_per_kg', 0),
                    "cost_contribution_per_ton": round(cost_contrib * 1000, 2),
                    "is_synthetic": ing.get('is_synthetic_aa', False),
                    "price_source": ing.get('price_source', 'unknown')
                })
        
        # Sort by percentage (largest first)
        mix.sort(key=lambda x: x['percentage'], reverse=True)
        
        # Calculate ratios
        ca_to_p = (analysis['calcium_pct'] / analysis['sttd_phosphorus_pct'] 
                   if analysis['sttd_phosphorus_pct'] > 0 else 0)
        
        formatted_solutions.append({
            "strategy": sol['strategy'],
            "label": sol['label'],
            "description": sol['description'],
            "safety_margin": sol['safety_margin'],
            "mix": mix,
            
            # Cost info (always include, will be flagged in wrapper if estimated)
            "cost_per_ton": round(total_cost * 1000, 2),
            "cost_per_kg": round(total_cost, 4),
            "cost_per_pig_daily": round(total_cost * dmi_adjusted_kg, 3),
            "cost_per_pig_monthly": round(total_cost * dmi_adjusted_kg * 30, 2),
            
            # Feeding quantities
            "daily_feed_per_pig_kg": round(dmi_adjusted_kg, 2),
            "monthly_feed_per_pig_kg": round(dmi_adjusted_kg * 30, 1),
            
            # Nutritional analysis
            "analysis": {k: round(v, 2) for k, v in analysis.items()},
            "ca_to_p_ratio": round(ca_to_p, 2),
            "ratio_status": "✅ Healthy (1.1-1.35:1)" if 1.1 <= ca_to_p <= 1.35 else "⚠️ Check ratio",
            
            # Metadata
            "ingredient_count": len(mix),
            "iterations": result.nit
        })
    
    # ============================================================
    # RETURN SUCCESS RESPONSE
    # ============================================================
    
    return json.dumps({
        "status": "success",
        "optimization_mode": price_check['mode'],
        "mode_reason": price_check['reason'],
        "mode_message": price_check['message'],
        "solutions": formatted_solutions,
        "requirements_used": requirements,
        "phase": phase_name,
        "total_ingredients_available": len(ingredients),
        "phase_restricted_count": len(phase_restricted),
        "calculation_metadata": {
            "constraints_applied": len(A_ub),
            "constraint_types": constraint_names[:10]  # First 10 for brevity
        }
    }), 200, headers 
