"""
Validation script for simulation results
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

def validate_results(results_file):
    """
    Validate simulation results for physical consistency.
    
    Parameters:
    -----------
    results_file : str or Path
        Path to simulation results JSON file
    
    Returns:
    --------
    dict
        Validation report
    """
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data['simulation_results']
    metrics = data['performance_metrics']
    
    validation = {
        'timestamp': datetime.now().isoformat(),
        'file': str(results_file),
        'checks': [],
        'passed': 0,
        'failed': 0,
        'warnings': 0
    }
    
    # Check 1: Positive pressure
    pressures = results['pressure']
    if all(p > 0 for p in pressures):
        validation['checks'].append({
            'check': 'Positive pressure',
            'status': 'PASS',
            'message': f'All {len(pressures)} pressure values are positive',
            'details': f'Range: {min(pressures):.0f} - {max(pressures):.0f} psi'
        })
        validation['passed'] += 1
    else:
        validation['checks'].append({
            'check': 'Positive pressure',
            'status': 'FAIL',
            'message': 'Negative pressure detected',
            'details': f'Min pressure: {min(pressures):.0f} psi'
        })
        validation['failed'] += 1
    
    # Check 2: Saturation bounds
    oil_sat = results['saturations']['oil']
    water_sat = results['saturations']['water']
    gas_sat = results['saturations']['gas']
    
    oil_ok = all(0 <= s <= 1 for s in oil_sat)
    water_ok = all(0 <= s <= 1 for s in water_sat)
    gas_ok = all(0 <= s <= 1 for s in gas_sat)
    
    if oil_ok and water_ok and gas_ok:
        validation['checks'].append({
            'check': 'Saturation bounds',
            'status': 'PASS',
            'message': 'All saturations within [0, 1] bounds',
            'details': f'Oil: {min(oil_sat):.3f}-{max(oil_sat):.3f}, '
                      f'Water: {min(water_sat):.3f}-{max(water_sat):.3f}, '
                      f'Gas: {min(gas_sat):.3f}-{max(gas_sat):.3f}'
        })
        validation['passed'] += 1
    else:
        validation['checks'].append({
            'check': 'Saturation bounds',
            'status': 'FAIL',
            'message': 'Saturation outside [0, 1] bounds',
            'details': f'Oil: {min(oil_sat):.3f}-{max(oil_sat):.3f}, '
                      f'Water: {min(water_sat):.3f}-{max(water_sat):.3f}, '
                      f'Gas: {min(gas_sat):.3f}-{max(gas_sat):.3f}'
        })
        validation['failed'] += 1
    
    # Check 3: Saturation sum ≈ 1
    saturation_sums = []
    for i in range(len(oil_sat)):
        total = oil_sat[i] + water_sat[i] + gas_sat[i]
        saturation_sums.append(total)
    
    max_deviation = max(abs(1 - s) for s in saturation_sums)
    
    if max_deviation < 0.01:  # 1% tolerance
        validation['checks'].append({
            'check': 'Saturation sum',
            'status': 'PASS',
            'message': 'Saturation sum close to 1.0',
            'details': f'Max deviation: {max_deviation:.4f}'
        })
        validation['passed'] += 1
    elif max_deviation < 0.05:  # 5% tolerance
        validation['checks'].append({
            'check': 'Saturation sum',
            'status': 'WARNING',
            'message': 'Saturation sum deviation > 1%',
            'details': f'Max deviation: {max_deviation:.4f}'
        })
        validation['warnings'] += 1
    else:
        validation['checks'].append({
            'check': 'Saturation sum',
            'status': 'FAIL',
            'message': 'Saturation sum far from 1.0',
            'details': f'Max deviation: {max_deviation:.4f}'
        })
        validation['failed'] += 1
    
    # Check 4: Realistic VRR
    vrr = metrics['efficiency']['voidage_replacement_ratio']
    
    if 0.8 <= vrr <= 1.2:
        validation['checks'].append({
            'check': 'VRR realistic',
            'status': 'PASS',
            'message': 'VRR within typical range',
            'details': f'VRR = {vrr:.2f}'
        })
        validation['passed'] += 1
    elif 0.5 <= vrr <= 2.0:
        validation['checks'].append({
            'check': 'VRR realistic',
            'status': 'WARNING',
            'message': 'VRR outside typical but possible range',
            'details': f'VRR = {vrr:.2f}'
        })
        validation['warnings'] += 1
    else:
        validation['checks'].append({
            'check': 'VRR realistic',
            'status': 'FAIL',
            'message': 'VRR unrealistic',
            'details': f'VRR = {vrr:.2f}'
        })
        validation['failed'] += 1
    
    # Check 5: Recovery factor
    rf = metrics['production']['oil_recovery_factor_percent']
    
    if 15 <= rf <= 35:
        validation['checks'].append({
            'check': 'Recovery factor',
            'status': 'PASS',
            'message': 'Recovery factor typical for waterflood',
            'details': f'RF = {rf:.1f}%'
        })
        validation['passed'] += 1
    elif 5 <= rf <= 50:
        validation['checks'].append({
            'check': 'Recovery factor',
            'status': 'WARNING',
            'message': 'Recovery factor atypical but possible',
            'details': f'RF = {rf:.1f}%'
        })
        validation['warnings'] += 1
    else:
        validation['checks'].append({
            'check': 'Recovery factor',
            'status': 'FAIL',
            'message': 'Recovery factor unrealistic',
            'details': f'RF = {rf:.1f}%'
        })
        validation['failed'] += 1
    
    # Check 6: Production rates non-negative
    oil_rates = results['production']['oil']
    water_rates = results['production']['water']
    gas_rates = results['production']['gas']
    inj_rates = results['injection']['water']
    
    oil_positive = all(r >= 0 for r in oil_rates)
    water_positive = all(r >= 0 for r in water_rates)
    gas_positive = all(r >= 0 for r in gas_rates)
    inj_positive = all(r >= 0 for r in inj_rates)
    
    if oil_positive and water_positive and gas_positive and inj_positive:
        validation['checks'].append({
            'check': 'Positive rates',
            'status': 'PASS',
            'message': 'All production and injection rates non-negative',
            'details': f'Oil: {min(oil_rates):.1f}-{max(oil_rates):.1f} STB/d'
        })
        validation['passed'] += 1
    else:
        validation['checks'].append({
            'check': 'Positive rates',
            'status': 'FAIL',
            'message': 'Negative rates detected',
            'details': f'Oil min: {min(oil_rates):.1f}, Water min: {min(water_rates):.1f}'
        })
        validation['failed'] += 1
    
    # Determine overall status
    if validation['failed'] == 0:
        validation['overall_status'] = 'PASS'
    elif validation['failed'] <= 2:
        validation['overall_status'] = 'WARNING'
    else:
        validation['overall_status'] = 'FAIL'
    
    return validation

def print_validation_report(validation):
    """
    Print validation report in readable format.
    """
    print("=" * 70)
    print("SIMULATION RESULTS VALIDATION REPORT")
    print("=" * 70)
    print(f"File: {validation['file']}")
    print(f"Timestamp: {validation['timestamp']}")
    print(f"Overall Status: {validation['overall_status']}")
    print("-" * 70)
    
    for check in validation['checks']:
        status_icon = {
            'PASS': '✅',
            'WARNING': '⚠️',
            'FAIL': '❌'
        }.get(check['status'], '❓')
        
        print(f"{status_icon} {check['check']}:")
        print(f"    {check['message']}")
        if 'details' in check:
            print(f"    Details: {check['details']}")
        print()
    
    print("-" * 70)
    print(f"Summary: {validation['passed']} passed, "
          f"{validation['warnings']} warnings, "
          f"{validation['failed']} failed")
    print("=" * 70)
    
    return validation['overall_status']

def validate_latest_results():
    """
    Find and validate the latest simulation results.
    """
    results_dir = Path("results_final")
    
    if not results_dir.exists():
        print("No results directory found. Run simulation first.")
        return None
    
    # Find JSON files
    json_files = list(results_dir.glob("*.json"))
    
    if not json_files:
        print("No JSON results found.")
        return None
    
    # Get latest file
    latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
    
    print(f"Validating latest results: {latest_file.name}")
    print()
    
    validation = validate_results(latest_file)
    status = print_validation_report(validation)
    
    # Save validation report
    report_file = results_dir / f"validation_{Path(latest_file).stem}.json"
    with open(report_file, 'w') as f:
        json.dump(validation, f, indent=2)
    
    print(f"Validation report saved: {report_file}")
    
    return status

if __name__ == "__main__":
    validate_latest_results()
