# GF Kubo Numeric
Numerical Kubo observable computation using Green's functions, Streda terms, etc.  
Specifically, the "kubo" package in this project can glue together the full Green's function for an orbitronic bilayer system with a plane interface between two metals that breaks the translation invariance of the system perpendicular to the interface.  
All specifications making this an orbitronic system are contained in the Hamiltonian setup in the models subpackage. Hence, the tools in this package can easily be adapted for other bilayer physics that break translation invariance in one direction by simply replacing the sub-system and interface information in models/orbitronic.py by another model choice (see example).  
The observable operators for the model in question are also specified in the model files. 
