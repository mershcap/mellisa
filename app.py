from flask import Flask, request,jsonify,render_template
import numpy as np
import pickle
import os

#creating app name
app=Flask(__name__)

#function to load the model
def Load():
	return pickle.load(open('maize_model.pkl','rb'))

#loading defalut page
@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	n_model=Load()
	img=request.files['file']
	img = image.load_img(img, target_size=(200, 200))
	img_array = image.img_to_array(img)
	img_batch = np.expand_dims(img_array, axis=0)
	img_p= preprocess_input(img_batch)
	pred = n_model.predict(img_p)
	n=pred[0]
	prediction=''
	if(n[0]==1.0):
		prediction='blight'
	if(n[1]==1.0):
		prediction='rust'
	if(n[2]==1.0):
		prediction='gray'
	if(n[3]==1.0):
		prediction='healthy'
	return render_template('index.html',output='The plant is/has :{}'.format(prediction))

if __name__=='__main__':
	port=int(os.environ.get('PORT',5000))
	app.run(port=port,debug=True,use_reloader=False)
