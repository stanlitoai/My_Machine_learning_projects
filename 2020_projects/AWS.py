import boto3
import json

filename =  "sign.jpg"
bucket = "rekog-test2"

client = boto3.client("rekognition", "us-east-1")

response = client.detect_text(Image = {"S3Object":{"Bucket": bucket, "Name": filename}})

for text in response["TextDetections"]:
    print(text["DectectedText"] + ":" + str(text["Confidence"]))