from flask import Flask, jsonify, request
from sklearn.externals import joblib
import sklearn


#curl -d "{\"Medidas\":[[1,2,3,4]]}" -H "Content-Type: application/json" -X POST http://127.0.0.1:5000/predecir

app= Flask(__name__)

@app.route("/")
def home():
    return 'La pagina esta funcionando bien'

@app.route("/predecir", methods=["POST"])
def predecir():
    json=request.get_json(force=True)
    medidas=json['Medidas']
    clf=joblib.load('Modelo_Entrenado.pkl')
    prediccion=clf.predict(medidas)
    return 'Las medidas que diste corresponden a la clase {0}\n\n'.format(prediccion)

if __name__ == '__main__':
    app.run()



