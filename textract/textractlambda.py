import json
import boto3
from base64 import b64encode, b64decode

def lambda_handler(event, context):
    
    '''
    function to process the input from API gateway, pass to Textract, 
    and post process the response before returning predictions.  Note that the 
    asyncronous calls to textract only support PNG and JPEG file formats (other
    formats are supported in the asyncronous calls).  Other limitations:
    
    * Lambda functions are limited by payload of 6mb of data.  If you want to 
    analyze larger files, you should first upload the document to s3.
    * API gateway has a timeout of 29 seconds
    * Textract requires the blob of bytes to be less than 5 MB
    
    '''

    eventBody = json.loads(json.dumps(event))['body']
    #imgb64 = json.loads(eventBody)['Image']
    #print(imgb64)

    
    # if bytes
    
    # if s3 path, call textract with s3
    
    # if URL 
    
    # call textract
    textract = boto3.client('textract')
    
    # blob of base64 encoded image bytes
    response = textract.detect_document_text(
        Document={
            'Bytes': b64decode(eventBody)
        })
    
    # parse response from textract
    
    return {
        'statusCode': 200,
        'body': json.dumps(response)
    }
