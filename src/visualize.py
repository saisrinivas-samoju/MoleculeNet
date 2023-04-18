from typing import List, Optional, Union
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False


def visualize_2d(smiles: Union[str, List[str]], size: tuple = (400, 400), 
                 mols_per_row: int = 3, sub_img_size: tuple = (300, 300)):
    if isinstance(smiles, str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        img = Draw.MolToImage(mol, size=size)
        return img
    elif isinstance(smiles, list):
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        # Filter out None values (invalid SMILES)
        mols = [m for m in mols if m is not None]
        if not mols:
            raise ValueError("No valid SMILES strings provided")
        img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=sub_img_size)
        return img
    else:
        raise TypeError("smiles must be a string or list of strings")


def visualize_3d(smiles: str, width: int = 400, height: int = 350, 
                 show_labels: bool = True, label_hydrogens: bool = True, 
                 transparent_background: bool = True):
    if not PY3DMOL_AVAILABLE:
        raise ImportError(
            "py3Dmol is required for 3D visualization. "
            "Install it with: pip install py3Dmol"
        )
    
    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    # Generate 3D coordinates
    mol = Chem.AddHs(mol)  # Add hydrogens for better 3D structure
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)  # Generate 3D coordinates
        AllChem.MMFFOptimizeMolecule(mol)  # Optimize geometry
    except Exception as e:
        raise ValueError(f"Failed to generate 3D coordinates: {e}")
    
    # Convert to PDB format string
    pdb_block = Chem.MolToPDBBlock(mol)
    
    # Create py3Dmol viewer
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_block, 'pdb')
    view.setStyle({'stick': {}, 'sphere': {'radius': 0.3}})
    
    # Set background transparency
    if transparent_background:
        view.setBackgroundColor('transparent')
    
    # Add atom labels if requested
    if show_labels:
        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            element = atom.GetSymbol()
            
            # Skip hydrogens unless explicitly requested
            if not label_hydrogens and element == 'H':
                continue
            
            # Get 3D coordinates
            pos = conf.GetAtomPosition(atom_idx)
            x, y, z = pos.x, pos.y, pos.z
            
            # Add label with element symbol
            view.addLabel(
                element,
                {'position': {'x': x, 'y': y, 'z': z},
                 'font': 'Arial',
                 'fontSize': 12,
                 'fontColor': 'black',
                 'backgroundColor': 'white',
                 'backgroundOpacity': 0.7,
                 'borderThickness': 1,
                 'borderColor': 'black',
                 'borderRadius': 3}
            )
    
    view.zoomTo()
    return view


def visualize_3d_html(smiles: str, width: int = 400, height: int = 350, 
                      show_labels: bool = True, label_hydrogens: bool = True, 
                      transparent_background: bool = True, div_id: Optional[str] = None):
    if not PY3DMOL_AVAILABLE:
        raise ImportError(
            "py3Dmol is required for 3D visualization. "
            "Install it with: pip install py3Dmol"
        )
    
    import uuid
    
    # Generate unique ID if not provided
    if div_id is None:
        div_id = f"3dmolviewer_{uuid.uuid4().hex[:12]}"
    
    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    # Generate 3D coordinates
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception as e:
        raise ValueError(f"Failed to generate 3D coordinates: {e}")
    
    # Convert to PDB format string
    pdb_block = Chem.MolToPDBBlock(mol)
    
    # Escape PDB block for JavaScript string (escape backslashes, quotes, and newlines)
    # Need to escape: backslashes, double quotes, newlines
    pdb_block_escaped = (pdb_block
                        .replace('\\', '\\\\')  # Escape backslashes first
                        .replace('"', '\\"')    # Escape double quotes
                        .replace('\n', '\\n')   # Escape newlines
                        .replace('\r', '\\r'))  # Escape carriage returns
    
    # Build JavaScript code for labels
    label_js = ""
    if show_labels:
        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            element = atom.GetSymbol()
            
            # Skip hydrogens unless explicitly requested
            if not label_hydrogens and element == 'H':
                continue
            
            # Get 3D coordinates
            pos = conf.GetAtomPosition(atom_idx)
            x, y, z = pos.x, pos.y, pos.z
            
            # Escape element symbol for JavaScript
            element_escaped = element.replace('"', '\\"')
            label_js += f'\t\tviewer_{div_id}.addLabel("{element_escaped}",{{"position": {{"x": {x}, "y": {y}, "z": {z}}}, "font": "Arial", "fontSize": 12, "fontColor": "black", "backgroundColor": "white", "backgroundOpacity": 0.7, "borderThickness": 1, "borderColor": "black", "borderRadius": 3}});\n'
    
    # Build background color
    bg_color = '"transparent"' if transparent_background else '"white"'
    
    # Generate HTML following py3Dmol's pattern
    # Use 100% width to fill container, maintain aspect ratio with height
    html = f'''<div id="{div_id}" style="position: relative; width: 100%; height: {height}px; max-width: {width}px; margin: 0 auto;"></div>
<script>
var loadScriptAsync_{div_id} = function(uri){{
  return new Promise((resolve, reject) => {{
    var savedexports, savedmodule;
    if (typeof exports !== 'undefined') savedexports = exports;
    else exports = {{}}
    if (typeof module !== 'undefined') savedmodule = module;
    else module = {{}}

    var tag = document.createElement('script');
    tag.src = uri;
    tag.async = true;
    tag.onload = () => {{
        exports = savedexports;
        module = savedmodule;
        resolve();
    }};
    var firstScriptTag = document.getElementsByTagName('script')[0];
    firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);
  }});
}};

if(typeof $3Dmolpromise === 'undefined') {{
  $3Dmolpromise = null;
  $3Dmolpromise = loadScriptAsync_{div_id}('https://cdn.jsdelivr.net/npm/3dmol@2.5.3/build/3Dmol-min.js');
}}

var viewer_{div_id} = null;
$3Dmolpromise.then(function() {{
  try {{
    var container = document.getElementById("{div_id}");
    if (!container) {{
      console.error("Container {div_id} not found");
      return;
    }}
    viewer_{div_id} = $3Dmol.createViewer(container,{{backgroundColor:{bg_color}}});
    viewer_{div_id}.addModel("{pdb_block_escaped}","pdb");
    viewer_{div_id}.setStyle({{"stick": {{}}, "sphere": {{"radius": 0.3}}}});
{label_js}
    viewer_{div_id}.zoomTo();
    viewer_{div_id}.render();
  }} catch(error) {{
    console.error("Error initializing 3D viewer {div_id}:", error);
  }}
}}).catch(function(error) {{
  console.error("Error loading 3Dmol.js:", error);
}});
</script>'''
    
    return html

