-- Grid geometry for SPE9
-- Cartesian grid with 10-degree dip in x-direction

DX
9000*300 /

DY
9000*300 /

DZ
600*20  / -- Layer 1
600*15  / -- Layer 2
600*26  / -- Layer 3
600*15  / -- Layer 4
600*16  / -- Layer 5
600*14  / -- Layer 6
600*8   / -- Layer 7
600*8   / -- Layer 8
600*18  / -- Layer 9
600*12  / -- Layer 10
600*19  / -- Layer 11
600*18  / -- Layer 12
600*20  / -- Layer 13
600*50  / -- Layer 14
600*100 / -- Layer 15

-- TOPS will be included from external file
INCLUDE
'TOPSVALUES.DATA' /

-- Permeability will be included from external file
PERMX
INCLUDE
'PERMVALUES.DATA' /

-- Set permeability in Y and Z directions
COPY
PERMX PERMY /
PERMX PERMZ /
/

-- Vertical permeability is 1% of horizontal
MULTIPLY
PERMZ 0.01 /
/
