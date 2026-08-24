import flask
import requests
import py3Dmol
from rdkit import Chem
from rdkit.Chem import Draw

compound_id = "NCGC00178831-03"
url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{compound_id}/property/IsomericSMILES/TXT"

response = requests.get(url)
if response.status_code == 200:
    smiles = response.text.strip()
    print(f"SMILES for {compound_id}: {smiles}")
else:
    print(f"Error retrieving SMILES for {compound_id}")



app = flask.Flask(__name__)

@app.route("/")
def index():

    molecule = Chem.MolFromSmiles(smiles)
    Draw.MolToImage(molecule)

    # draw in 3d
    import py3Dmol
    view = py3Dmol.view()
    view.addModel(Chem.MolToMolBlock(molecule), 'sdf')
    view.setStyle({'stick': {}})
    
    return view.show()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9980)

