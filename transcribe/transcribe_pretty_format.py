import argparse, os
import numpy as np
import json
import datetime
import boto3

def format_transcript(json_transcript_uri,output_filename):
    
    temp = []

    if 's3://' in json_transcript_uri:
        s3 = boto3.resource('s3')
        bucket = json_transcript_uri.split('/')[3]
        key = '/'.join(json_transcript_uri.split('/')[4:])
        
        # read file
        obj = s3.Object(bucket, key)
        file_content = obj.get()['Body'].read().decode('utf-8')
    else:
        f = open(json_transcript_uri,'r')
        file_content = f.read()
        f.close()

    data = json.loads(file_content)
    raw_transcript = data['results']['transcripts'][0]['transcript']
    
    with open(output_filename,'w') as w:
        
        # check type of transcript - channel vs speaker 
        if 'channel_labels' in data['results'].keys():
            labels = data['results']['channel_labels']['channels']
            var = 'channel_label'

        elif 'speaker_labels' in data['results'].keys():
            labels = data['results']['speaker_labels']['segments']
            var = 'speaker_label'

        else:
            'Not channel or speaker encoded'
            
        # get speaker times
        speaker_start_times={}
        for label in labels:
            for item in label['items']:
                start_time = item.get("start_time", "")
                if start_time != "":
                    speaker_start_times[item['start_time']] = label[var]
                    
        # get text per speaker
        items = data['results']['items']
        lines, line, time = [], '', 0
        speaker, i = 'null', 0
        for item in items:
            i=i+1
            content = item['alternatives'][0]['content']
            if item.get('start_time'):
                current_speaker=speaker_start_times[item['start_time']]
            elif item['type'] == 'punctuation':
                line = line+content
            if current_speaker != speaker:
                if speaker:
                    lines.append({'speaker':speaker, 'line':line, 'time':time})
                line=content
                speaker=current_speaker
                time=item['start_time']
            elif item['type'] != 'punctuation':
                line = line + ' ' + content
        lines.append({'speaker':speaker, 'line':line,'time':time})
        
        # output to file
        sorted_lines = sorted(lines,key=lambda k: float(k['time']))
        for line_data in sorted_lines:
            if line_data.get('speaker') != 'null':
                line='[' + str(datetime.timedelta(seconds=int(round(float(line_data['time']))))) + '] ' + line_data.get('speaker') + ': ' + line_data.get('line')
                w.write(line + '\n\n')
    
    return sorted_lines

if __name__ == "__main__":
  
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-filename', type=str, default='temp.txt')
    parser.add_argument('--input-filename', type=str)
    args, _ = parser.parse_known_args()
    output_filename = args.output_filename
    input_filename = args.input_filename

    # format transcript
    _ = format_transcript(input_filename,output_filename)


    
    