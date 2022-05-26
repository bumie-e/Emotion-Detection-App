import streamlit as st
import requests
import cv2, operator, os, time
from PIL import Image
from base64 import decodebytes
import numpy as np
from dotenv import load_dotenv
load_dotenv()























# Load the values from .env
key = os.environ['KEY']
endpoint = os.environ['ENDPOINT']
headers = {'Ocp-Apim-Subscription-Key': key}

# Add Parameters
face_api_url = endpoint + '/face/v1.0/detect?detectionModel=detection_01&returnFaceId=true&returnFaceLandmarks=false&returnFaceAttributes=emotion' 
params = {
    'detectionModel': 'detection_03',
    'returnFaceId': 'true'
}



# Add Image Url
image_url = 'https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/ComputerVision/Images/faces.jpg'


















##########
##### Set up sidebar.
##########


#st.sidebar.write('Add an Image Link')
link = st.sidebar.text_input('Add an Image Link', image_url)
image = Image.open(requests.get(link, stream=True).raw)
st.sidebar.image(image,
                 use_column_width=True)

## Title.
st.write('# Emotion Detection App')

## Subtitle.
st.write('### Image')


# Function to make API request        
def processRequest( json, data, headers, params ):
    """

    Parameters:
    json: Used when processing images from its URL. See API Documentation
    data: Used when processing image read from disk. See API Documentation
    headers: Used to pass the key information and the data type request
    """

    retries = 0
    result = None
    _maxNumRetries = 10

    while True:

        response = requests.request( 'post', face_api_url, json = json, data = data, headers = headers, params = params )

        if response.status_code == 429: 

            print( "Message: %s" % ( response.json()['error']['message'] ) )

            if retries <= _maxNumRetries: 
                time.sleep(1) 
                retries += 1
                continue
            else: 
                print( 'Error: failed after retrying!' )
                break

        elif response.status_code == 200 or response.status_code == 201:

            if 'content-length' in response.headers and int(response.headers['content-length']) == 0: 
                result = None 
            elif 'content-type' in response.headers and isinstance(response.headers['content-type'], str):
                if 'application/json' in response.headers['content-type'].lower(): 
                    result = response.json() if response.content else None 
                elif 'image' in response.headers['content-type'].lower(): 
                    result = response.content
        else:
            print( "Error code: %d" % ( response.status_code ) )
            print( "Message: %s" % ( response.json()['error']['message'] ) )

        break

# Function to show rectangle on Image
def renderResultOnImage( result, img ):
    
    """Display the obtained results onto the input image"""
    
    for currFace in result:
        faceRectangle = currFace['faceRectangle']
    
        cv2.rectangle( img,(faceRectangle['left'],faceRectangle['top']),
                           (faceRectangle['left']+faceRectangle['width'], faceRectangle['top'] + faceRectangle['height']),
                       color = (255,0,0), thickness = 2 )


    for currFace in result:
        faceRectangle = currFace['faceRectangle']
        currEmotion = max(currFace['faceAttributes']['emotion'].items(), key=operator.itemgetter(1))[0]


        textToWrite = "%s" % ( currEmotion )
        cv2.putText( img, textToWrite, (faceRectangle['left'],faceRectangle['top']-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1 )
  
    
















## POST to the API.
response = requests.post(face_api_url, params=params,
                        headers=headers, json={"url": link})
if response is not None:
    # Load the original image, fetched from the URL
    arr = np.asarray( bytearray( requests.get( link ).content ), dtype=np.uint8 )
    img = cv2.cvtColor( cv2.imdecode( arr, -1 ), cv2.COLOR_BGR2RGB )
    renderResultOnImage(response.json(), img )
    st.image(img,
            use_column_width=True)