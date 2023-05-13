import os
import json
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from itertools import groupby

# from email.mime.base import MIMEBase
# from email.mime.text import MIMEText
# from email.utils import COMMASPACE
# from email import encoders
from flask import Flask, render_template, request, send_file, url_for, redirect
# from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from pipelines import pipeline  # Replace with your own function for generating answers
from question_generation import clean_text
# from huggingface_hub import login
# access_token = "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"
# login(token = access_token)
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

access_token = "hf_..."


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/answer", methods=["POST"])
def answer():
    passage = request.form["passage"]
    # cleaned_passage = clean_text(passage)
    # question = request.form["question"]
    # model = AutoModel.from_pretrained("valhalla/t5-small-qg-prepend", use_auth_token='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')
    # nlp = pipeline('question-generation', model='valhalla/t5-small-qg-prepend', qg_format="prepend", use_auth_token='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')
    # nlp = pipeline("question-answering", model = "distilbert-base-cased-distilled-squad", tokenizer = "bert-base-cased")
    # model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-small-e2e-qg", use_auth_token='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')
    # tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-small-e2e-qg")
    # nlp = pipeline("e2e-qg", use_auth_token='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')
    # nlp = pipeline("multitask-qa-qg", use_auth_token='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')

    # tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-small-qa-qg-hl", use_auth_token='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')
    #
    # model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-small-qa-qg-hl", use_auth_token='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')
    # tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-small-qg-hl", use_auth_token='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')

    # model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-small-qg-prepend", use_auth_token='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')

    # nlp = pipeline("question-generation", use_auth_token='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')
    nlp = pipeline("question-generation", model="valhalla/t5-small-qg-prepend", qg_format="prepend", use_auth_token='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')
    # nlp = pipeline("multitask-qa-qg", model="valhalla/distilt5-qa-qg-hl-6-4")
    # qg_model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5_squad_v1", use_auth_token='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')
    # qg_tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5_squad_v1", use_auth_token='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')
    # qa_model = pipeline("question-answering", model="mrm8488/t5-small-finetuned-squad-qgen",
    #                     tokenizer="t5-base", use_auth_token='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')
    # print(passage)
    answer = nlp(passage)

    answer = [{'question': key, 'answer': max(item['answer'] for item in values)}
                     for key, values in groupby(answer, lambda dct: dct['question'])]

    for x in answer:
        if "<pad> " in x['answer']:
            x['answer'] = x['answer'].replace("<pad> ", "")
    selected_task = request.form["task"]
    question_list = []
    # Use the selected model
    if selected_task == "single-qa":
        res = []
        for x in answer:
            answer_task1 = x
            answer = answer_task1
            res = answer
            break

    if selected_task == "multi-qg":
        nlp = pipeline("e2e-qg", use_auth_token='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')
        answer = nlp(passage)
        res = []
        [res.append(x) for x in answer if x not in res]
        # for x in answer:
        #     question_list.append(x['question'])
        #     answer = question_list
    if selected_task == "e2e-qa":
        answer = [{'question': key, 'answer': max(item['answer'] for item in values)}
                     for key, values in groupby(answer, lambda dct: dct['question'])]
        res = []
        [res.append(x) for x in answer if x not in res]

    with open('answer.json', 'w') as outfile:
        json.dump(answer, outfile)
        # json.dumps(qa_pairs, indent=4)
    # return render_template('answer.html', passage=cleaned_passage, question=question, answer=answer)
    return render_template("answer.html", answer=res)


@app.route('/download')
def download():
    return send_file('answer.json', as_attachment=True)


@app.route('/email', methods=['POST'])
def email_answer():
    email = request.form['email']
    # passage = request.form['passage']
    # question = request.form['question']
    answer = request.form['answer']

    # Create a JSON file for the answer data
    answer_data = {
        # "passage": passage,
        # "question": question,
        "answer": answer
    }
    with open('answer.json', 'w') as f:
        json.dump(answer_data, f)

    # Create a MIME message
    msg = MIMEMultipart()
    msg['From'] = 'lahiri.aritra@gmail.com'
    msg['To'] = email
    msg['Subject'] = 'Answer to your question'

    # Attach the JSON file to the email
    with open('answer.json', 'rb') as f:
        attachment = MIMEApplication(f.read(), _subtype='json')
        attachment.add_header('content-disposition', 'attachment', filename='answer.json')
        msg.attach(attachment)

    # Send the email
    with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()

        smtp.login('lahiri.aritra@gmail.com', 'vvhktmjpoxmmrdsy')
        smtp.send_message(msg)

    # Delete the JSON file after sending the email
    os.remove('answer.json')

    return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(debug=True)
