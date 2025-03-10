import json
import logging
import base64
import boto3
import mimetypes
from requests_toolbelt.multipart import decoder

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize the SageMaker runtime client
sagemaker_runtime = boto3.client('sagemaker-runtime')

# Define the SageMaker endpoint name
SAGEMAKER_ENDPOINT = 'image-classification-endpoint'

def lambda_handler(event, context):
    try:
        logger.info("Lambda function invoked...")

        # Extract request body and headers
        body = event.get('body', '')
        is_base64_encoded = event.get('isBase64Encoded', False)
        headers = event.get('headers', {})
        content_type = headers.get('Content-Type', headers.get('content-type', ''))

        # Decode if base64-encoded
        if is_base64_encoded:
            body = base64.b64decode(body)

        # Parse multipart form-data
        multipart_data = decoder.MultipartDecoder(body, content_type)

        image_data = None
        inferred_content_type = 'application/octet-stream'

        # Look for the part with the key 'image'
        for part in multipart_data.parts:
            content_disposition = part.headers.get(b'Content-Disposition', b'').decode()
            if 'name="image"' in content_disposition:
                image_data = part.content
                
                # Infer content type dynamically
                filename = content_disposition.split('filename="')[-1].strip('"') if 'filename="' in content_disposition else None
                if filename:
                    inferred_content_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
                break

        if not image_data:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'No image file provided in the request.'})
            }

        # Invoke the SageMaker endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            ContentType=inferred_content_type,  # Dynamically detected MIME type
            Body=image_data
        )

        # Parse the response from SageMaker
        result = json.loads(response['Body'].read().decode())

        # Extract classification result
        predicted_label = result.get('predicted_label', 'Unknown')

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({'classification': predicted_label})
        }

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({'error': str(e)})
        }
