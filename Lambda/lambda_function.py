import json
import boto3
import email
from sms_spam_classifier_utilities import one_hot_encode
from sms_spam_classifier_utilities import vectorize_sequences

def lambda_handler(event, context):
    # bucket = 'mailstoragebt'
    bucket = event['Records'][0]['s3']['bucket']['name']
    # key ='qpa98uavokrk11b0rp30lmkbtc54seimu1nvrgo1'
    key = event['Records'][0]['s3']['object']['key']
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=key)
    
    emailobj = email.message_from_bytes(response['Body'].read())
    to_email = emailobj.get('To')
    from_email = emailobj.get('From')
    from_email = from_email[from_email.find("<")+1:from_email.find(">")]
    date = emailobj.get('Date')
    subject = emailobj.get('Subject')
    body = emailobj.get_payload()[0].get_payload()
    
    endpoint = 'sms-spam-classifier-mxnet-2021-04-22-15-34-38-780'
    runtime = boto3.client('runtime.sagemaker','us-east-1')
    vocabulary_length = 9013
    
    input_mail = [body.strip()]
    onehot_input = one_hot_encode(input_mail, vocabulary_length)
    encoded_input = vectorize_sequences(onehot_input, vocabulary_length)
    data = json.dumps(encoded_input.tolist())
    response = runtime.invoke_endpoint(EndpointName=endpoint, ContentType='application/json',Body=data)
    res = json.loads(response["Body"].read())
    label = 'Ham' if res['predicted_label'][0][0] == 0 else "Spam"
    score = round(res['predicted_probability'][0][0],4)
    print(res)
    
    message = "We received your email sent at {0} with the subject {1}.\nHere \
is a 240 character sample of the email body:\n\n{2}\nThe email was \
categorized as {3} with a {4}% confidence.".format(to_email,subject,body,label,score)
    
    
    client = boto3.client('ses','us-east-1')
    response = client.send_email(
            Destination={
                'ToAddresses': [
                    from_email
                ],
            },
            Message={
                'Body': {
                    'Text': {
                        'Charset': 'UTF-8',
                        'Data': message,
                    },
                },
                'Subject': {
                    'Charset': 'UTF-8',
                    'Data': 'Spam report',
                },
            },
            Source= to_email,
        )
