import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))


input_data = [7.0, 100.0, 200.0, 0.0, 300.0, 0.0, 0.0, 0.0, 5.0]
input_as_array = np.array(input_data)
reshaped_input = input_as_array.reshape(1,-1)
prediction = model.predict(reshaped_input)
print(prediction)
    
if (prediction[0]== 0):
    print('The water is not potable.')
else:
    print('The water is potable.')