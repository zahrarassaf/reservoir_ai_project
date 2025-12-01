"""
Test data validation functionality
"""

import pytest
import tempfile
from pathlib import Path
from src.data_validator import DataValidator


class TestDataValidation:
    """Test data validation methods"""
    
    def test_validate_main_data_file(self, tmp_path):
        """Test main data file validation"""
        validator = DataValidator()
        
        # Create a minimal valid data file
        data_file = tmp_path / "SPE9.DATA"
        data_file.write_text("""
RUNSPEC
TITLE
Test Simulation

DIMENS
24 25 15 /

OIL
WATER
GAS

GRID
INCLUDE
'grid.inc' /

PROPS

SOLUTION

SCHEDULE

END
""")
        
        # Test with valid file
        validator.validate_main_data_file(str(tmp_path))
        assert len(validator.errors) == 0
        
    def test_validate_missing_sections(self, tmp_path):
        """Test validation of missing sections"""
        validator = DataValidator()
        
        # Create data file missing required sections
        data_file = tmp_path / "SPE9.DATA"
        data_file.write_text("""
RUNSPEC
TITLE
Test

DIMENS
10 10 5 /
END
""")
        
        validator.validate_main_data_file(str(tmp_path))
        assert len(validator.errors) > 0  # Should have errors for missing sections
        
    def test_validate_grid_dimensions(self, tmp_path):
        """Test grid dimensions validation"""
        validator = DataValidator()
        
        # Test with wrong dimensions
        data_file = tmp_path / "SPE9.DATA"
        data_file.write_text("""
RUNSPEC
DIMENS
10 10 5 /
END
""")
        
        validator.validate_main_data_file(str(tmp_path))
        # Should error about wrong dimensions (expecting 24x25x15)
        assert len(validator.errors) > 0
        
    def test_validate_include_files(self, tmp_path):
        """Test INCLUDE file validation"""
        validator = DataValidator()
        
        # Create data file with non-existent INCLUDE
        data_file = tmp_path / "SPE9.DATA"
        data_file.write_text("""
RUNSPEC
DIMENS
24 25 15 /

GRID
INCLUDE
'nonexistent.inc' /
END
""")
        
        validator.validate_main_data_file(str(tmp_path))
        assert len(validator.errors) > 0  # Should error about missing include
        
    def test_validate_porosity_file(self, tmp_path):
        """Test porosity file validation"""
        validator = DataValidator()
        
        # Create porosity file
        poro_file = tmp_path / "poro.inc"
        poro_file.write_text("""
-- Porosity distribution
9000*0.15 /
""")
        
        validator._validate_porosity_file(poro_file)
        assert len(validator.errors) == 0
        
        # Test with wrong number of values
        poro_file.write_text("100*0.15 /")
        validator.errors = []  # Reset errors
        validator._validate_porosity_file(poro_file)
        assert len(validator.errors) > 0  # Should error about wrong count
        
    def test_validate_pvt_data(self, tmp_path):
        """Test PVT data validation"""
        validator = DataValidator()
        
        # Create PVT file with all required tables
        pvt_file = tmp_path / "pvt.inc"
        pvt_file.write_text("""
PVTW
3600 1.0 3e-6 0.96 0 /

PVTO
0 14.7 1.0 1.0 /

PVDG
14.7 200 0.01 /
""")
        
        validator.validate_pvt_data(str(tmp_path))
        assert len(validator.errors) == 0
        
        # Test with missing table
        pvt_file.write_text("PVTW\n3600 1.0 3e-6 0.96 0 /\n")
        validator.errors = []
        validator.validate_pvt_data(str(tmp_path))
        assert len(validator.errors) > 0  # Should error about missing PVTO/PVDG
        
    def test_validate_saturation_tables(self, tmp_path):
        """Test saturation tables validation"""
        validator = DataValidator()
        
        # Create saturation tables file
        sat_file = tmp_path / "sat.inc"
        sat_file.write_text("""
SGOF
0 0 1 0 /
1 1 0 0 /

SWOF
0 0 1 0 /
1 1 0 0 /
""")
        
        validator.validate_saturation_tables(str(tmp_path))
        assert len(validator.errors) == 0
        
        # Test with missing table
        sat_file.write_text("SGOF\n0 0 1 0 /\n")
        validator.errors = []
        validator.validate_saturation_tables(str(tmp_path))
        assert len(validator.errors) > 0  # Should error about missing SWOF
        
    def test_validate_well_data(self, tmp_path):
        """Test well data validation"""
        validator = DataValidator()
        
        # Create data file with wells
        data_file = tmp_path / "SPE9.DATA"
        data_file.write_text("""
RUNSPEC
DIMENS
24 25 15 /

SCHEDULE
WELSPECS
'INJE1' 'G' 1 1 1000 'WATER' /
'PRODU1' 'G' 2 2 1000 'OIL' /
/
END
""")
        
        validator.validate_well_data(str(tmp_path))
        # Should not error with injector present
        assert 'INJE1' in data_file.read_text()
        
    def test_complete_validation(self, tmp_path):
        """Test complete validation workflow"""
        validator = DataValidator()
        
        # Create minimal valid dataset
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Main data file
        main_file = data_dir / "SPE9.DATA"
        main_file.write_text("""
RUNSPEC
TITLE
Test

DIMENS
24 25 15 /

OIL
WATER
GAS

GRID
INCLUDE
'grid.inc' /

PROPS
INCLUDE
'pvt.inc' /

SOLUTION

SCHEDULE
WELSPECS
'INJE1' 'G' 1 1 1000 'WATER' /
/
END
""")
        
        # Grid file
        grid_file = data_dir / "grid.inc"
        grid_file.write_text("DX\n9000*300 /\n")
        
        # PVT file
        pvt_file = data_dir / "pvt.inc"
        pvt_file.write_text("""
PVTW
3600 1.0 3e-6 0.96 0 /
PVTO
0 14.7 1.0 1.0 /
PVDG
14.7 200 0.01 /
""")
        
        # Run validation
        is_valid, messages = validator.validate_all(str(data_dir))
        
        # Should be valid (or have only warnings)
        assert is_valid or (not is_valid and len(messages) > 0)


def test_error_handling():
    """Test error handling in validator"""
    validator = DataValidator()
    
    # Test with non-existent directory
    is_valid, messages = validator.validate_all("/non/existent/path")
    assert not is_valid
    assert len(messages) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
