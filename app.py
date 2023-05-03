import os
import json
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
# from email.mime.base import MIMEBase
# from email.mime.text import MIMEText
# from email.utils import COMMASPACE
# from email import encoders
from flask import Flask, render_template, request, send_file, url_for, redirect
from pipelines import pipeline # Replace with your own function for generating answers
from question_generation import clean_text

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/answer", methods=["POST"])
def answer():
    passage = request.form["passage"]
    cleaned_passage = clean_text(passage)
    # question = request.form["question"]
    nlp = pipeline("question-generation", model="valhalla/t5-small-qg-prepend", qg_format="prepend")
    # nlp = pipeline("question-generation")
    # nlp = pipeline("multitask-qa-qg", model="valhalla/distilt5-qa-qg-hl-6-4")
    answer = nlp(cleaned_passage)
    for x in answer:
        if "<pad> " in x['answer']:
            x['answer'] = x['answer'].replace("<pad> ", "")
    with open('answer.json', 'w') as outfile:
        json.dump(answer, outfile)
    # return render_template('answer.html', passage=cleaned_passage, question=question, answer=answer)
    return render_template("answer.html", passage=passage, answer=answer)

@app.route('/download')
def download():
    return send_file('answer.json', as_attachment=True)


@app.route('/email', methods=['POST'])
def email_answer():
    email = request.form['email']
    passage = request.form['passage']
    # question = request.form['question']
    answer = request.form['answer']

    # Create a JSON file for the answer data
    answer_data = {
        "passage": passage,
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