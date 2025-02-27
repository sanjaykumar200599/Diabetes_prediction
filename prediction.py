#This is an example code you can use it.
# import numpy as np
# import pickle
# import streamlit as st

# #loading the model
# loaded_model = pickle.load(open('trained_model.sav','rb'))

# #creating the function or prediction

# def diabetes_prediction(input_data):
#     input_data_as_numpy_array = np.asarray(input_data)

#     input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#     prediction = loaded_model.predict(input_data_reshaped)
#     print(prediction)

#     if (prediction[0] == 0):
#         return 'The person is not diabetic'
#     else:
#         return 'The person is diabetic'

# def main():

#     #Giving the title
#     st.title('Diabetes Prediction App')

#     #Getting the input data from the user
#     # Example: Ensure input values are properly converted
    


#     Pregnancies=st.text_input('Number of Pregnancies')
#     Glucose=st.text_input('Glucose Level')
#     BloodPressure=st.text_input('Blood Pressure Value')
#     SkinThickness=st.text_input('Skin Thickness Value')
#     Insulin=st.text_input('Insulin Level')
#     BMI=st.text_input('BMI Level')
#     DiabetesPedigreeFunction=st.text_input('Diabetes Pedigree Function Value')
#     Age=st.text_input('Age')
   
#    
#     diagnosis=''

#     
#     if(st.button('Diabetes Test Result')):
#         diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
#         st.success(diagnosis)
    

# if __name__ == '__main__':
#     main()

import streamlit as st
import numpy as np
import pickle

# Load your trained model
loaded_model = pickle.load(open('trained_model.sav','rb'))

def diabetes_prediction(input_data):
    input_data_reshaped = np.array(input_data, dtype=np.float64).reshape(1, -1)  # Ensure proper format
    prediction = loaded_model.predict(input_data_reshaped)
    return prediction

def main():
    st.title("Diabetes Prediction App")

    # Get user input
    Pregnancies = st.number_input("Pregnancies")
    Glucose = st.number_input("Glucose")
    BloodPressure = st.number_input("Blood Pressure")
    SkinThickness = st.number_input("Skin Thickness")
    Insulin = st.number_input("Insulin")
    BMI = st.number_input("BMI")
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function")
    Age = st.number_input("Age")


    if st.button("Predict"):
        try:
            # Ensure all values are properly converted
            input_values = [
                float(Pregnancies), float(Glucose), float(BloodPressure),
                float(SkinThickness), float(Insulin), float(BMI),
                float(DiabetesPedigreeFunction), float(Age)
            ]

            # Run prediction
            diagnosis = diabetes_prediction(input_values)

            # Display result
            st.success(f"Prediction Result: {'The person is Diabetic' if diagnosis[0] == 1 else 'The Person is not Diabetic'}")

        except ValueError:
            st.error("Invalid input! Please enter numerical values.")

if __name__ == "__main__":
    main()


    
