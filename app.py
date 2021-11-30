from flask import Flask, request,jsonify,render_template
import numpy as np
import pickle
import os

#creating app name
app=Flask(__name__)

#function to load the model
def Load():
	return pickle.load(open('player_rating.pkl','rb'))

#loading defalut page
@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	gk1=float(request.form.get("gk1",""))
	gk2=float(request.form.get("gk2",""))
	gk3=float(request.form.get("gk3",""))
	gk4=float(request.form.get("gk4",""))
	gk5=float(request.form.get("gk5",""))
	gk6=float(request.form.get("gk6",""))
	gk7=float(request.form.get("gk7",""))
	gk8=float(request.form.get("gk8",""))
	gk9=float(request.form.get("gk9",""))
	gk10=float(request.form.get("gk10",""))
	features=[gk1,gk2,gk3,gk4,gk5,gk6,gk7,gk8,gk9,gk10]
	values=[np.array(features)]
	print(values)
	model=Load()
	y_pred=model.predict(values)
	return render_template('index.html',output='The player predicted score is:{}'.format(y_pred))

if __name__=='__main__':
	port=int(os.environ.get('PORT',5000))
	app.run(port=port,debug=True,use_reloader=False)
