from ultralytics import YOLO
import pathlib
import torch
import cv2
import numpy as np
import pandas as pd
import fitz
from pdf2image.pdf2image import convert_from_path
from transformers import pipeline
import spacy
from random import randint
import re
from flask import Flask, request
import requests
import pymongo
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import os
from flask_cors import CORS
from flask_ngrok import run_with_ngrok


# All model loading part
pg = 0
bbox_model = YOLO("block_best.pt")
bbox_model.iou = 0.85  # NMS IoU threshold
bbox_model.agnostic = False
nlp_best = spacy.load("model-best")
block_classifier = pipeline(
    "text-classification",
    model="Resume_Sections",
    truncation=True,
)

conn = pymongo.MongoClient(
    "mongodb+srv://shivamthakkar:shivam.thakkar123@resumeparser.iwpgki3.mongodb.net/"
)

db = conn.test
resume = db.Testing1


def classify_text(block):
    classification = block_classifier(block)
    pred_class = classification[0]["label"]
    return pred_class


def get_ner(text, section):
    doc = nlp_best(text)
    entities = []
    if section == "education" or section == "professional_experiences":
        for ent in doc.ents:
            if ent.label_.lower() != "location":
                if ent.text != "EDUCATION":
                    entities.append((ent.label_.lower(), ent.text))
    else:
        entities = {}
        for ent in doc.ents:
            if ent.label_.lower() not in entities:
                entities[ent.label_.lower()] = [ent.text]
            else:
                entities[ent.label_.lower()].append(ent.text)
    return entities


def getbboxtext(img, image_height, image_width, page) -> list:
    skills = []
    projects = ""
    education_list = []
    experience_list = []
    achievements = ""
    urls = []
    name = ""
    email = ""
    phone = ""
    img = cv2.resize(img, (round(image_width), round(image_height)))
    results = bbox_model.predict(img)[0]
    boxes = results.boxes
    resume_text = ""
    for items in boxes.xyxy:
        x1, y1, x2, y2 = items.tolist()
        annot = page.draw_rect([x1, y1, x2, y2], width=1)
        extracted_text = page.get_textbox([x1, y1, x2, y2])
        resume_text += extracted_text
        predicted_class = classify_text(extracted_text)
        urls += re.findall(r"(https?://\S+)", extracted_text)
        if predicted_class == "skills":
            temp_text = extracted_text.replace("\n", ",")
            skills += temp_text.split(",")
        elif predicted_class == "contact/name/title":
            ner_details = get_ner(extracted_text, "contact/name/title")
            try:
                name = ner_details["person"]
            except:
                pass
            try:
                email = ner_details["email"]
            except:
                pass
            try:
                phone = ner_details["contact"]
            except:
                pass
        elif predicted_class == "projects":
            projects += extracted_text
        elif predicted_class == "education":
            ner_education = get_ner(extracted_text, "education")
            print(ner_education)
            education_st = []
            edu_st_keys = []
            for exp_ed in ner_education:
                if exp_ed[0] not in edu_st_keys:
                    education_st.append(exp_ed)
                    edu_st_keys.append(exp_ed[0])
                else:
                    degree = ""
                    institution = ""
                    edu_duration = ""
                    for i in education_st:
                        if i[0] == "education":
                            degree = i[1]
                        elif i[0] == "institution":
                            institution = i[1]
                        elif i[0] == "duration":
                            edu_duration = i[1]
                    education_list.append([degree, institution, edu_duration])
                    education_st = [exp_ed]
                    edu_st_keys = [exp_ed[0]]
            if len(education_st) != 0:
                degree = ""
                institution = ""
                edu_duration = ""
                for i in education_st:
                    if i[0] == "education":
                        degree = i[1]
                    elif i[0] == "institution":
                        institution = i[1]
                    elif i[0] == "duration":
                        edu_duration = i[1]
                education_list.append([degree, institution, edu_duration])
        elif predicted_class == "professional_experiences":
            ner_experience = get_ner(extracted_text, "professional_experiences")
            print(ner_experience)
            st = []
            st_keys = []
            for exp in ner_experience:
                if exp[0] not in st_keys:
                    st.append(exp)
                    st_keys.append(exp[0])
                else:
                    role = ""
                    company = ""
                    duration = ""
                    for i in st:
                        if i[0] == "role":
                            role = i[1]
                        elif i[0] == "organisation":
                            company = i[1]
                        elif i[0] == "duration":
                            duration = i[1]
                    experience_list.append([role, company, duration])
                    st = [exp]
                    st_keys = [exp[0]]
            if len(st) != 0:
                role = ""
                company = ""
                duration = ""
                for i in st:
                    if i[0] == "role":
                        role = i[1]
                    elif i[0] == "organisation":
                        company = i[1]
                    elif i[0] == "duration":
                        duration = i[1]
                experience_list.append([role, company, duration])

        elif predicted_class == "certificates":
            achievements += extracted_text
        else:
            pass
    # resume_text = " ".join(text)
    synopsis = "Your Candidate {0} is a potential job candidate for your company with a qualification of {1}. The candidate is well versed with the following skills: {2}. Here are the contact details :- Phone : {3} and Email : {4}".format(
        name,
        "Hello",
        (" ,".join(skills)),
        phone,
        email,
    )
    print(experience_list, education_list)
    return {
        # "Resume_id": resume_id,
        "Resume_data": resume_text,
        "Name": name,
        "Email": email,
        "Phone": phone,
        "Education": education_list,
        "Skills": skills,
        "Experience": experience_list,
        "Urls": urls,
        "Synopsis": synopsis,
        "Projects": projects,
        "Achievements": achievements,
    }


def matching(skills_str, skills_taken):
    skills = skills_taken
    count = 0
    skills_list = skills_str.split(",")
    skills_list = [i.strip(" ").lower() for i in skills_list]
    length = len(skills_list)
    skills = [i.lower() for i in skills]
    final = set(skills_list).intersection(set(skills))
    try:
        percentage = round(((len(final) / length) * 100))
    except:
        percentage = 0
    return percentage


app = Flask(_name_)
CORS(app)
app.config["DEBUG"] = True
app.config["ENV"] = "development"


@app.route("/add", methods=["POST"])
def upload():
    try:
        resume.drop()
    except:
        pass
    data = request.json
    urls = data["urls"]
    for url in urls:
        r = requests.get(
            url,
            stream=True,
        )
        url = url.split("?")
        ext = url[0].split(".")[-1]
        with open("%s.%s" % ("resume", ext), "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)  #
        doc = fitz.open("resume.pdf")
        page = doc[pg]
        image_width = page.rect.width
        image_height = page.rect.height
        pages = convert_from_path("resume.pdf")
        img = np.array(pages[0])
        final_output = getbboxtext(img, image_height, image_width, page)
        resume.insert_one(final_output)
        output_file = "boxes.pdf"
        doc.save(output_file, garbage=4, deflate=True, clean=True)
    return "Data Parsed"


@app.route("/getData", methods=["GET"])
def add():
    final = []
    data = resume.find()
    for d in data:
        try:
            del d["_id"]
        except:
            continue
        final.append(d)
    return {"data": final}


@app.route("/matching", methods=["POST"])
def matching_resume():
    final = []
    str_list = request.json
    str_list = str_list["skills"]
    data = resume.find()
    for d in data:
        temp = d.copy()
        percentage_match = matching(str_list, d["Skills"])
        final.append(percentage_match)
        d["match_percentage"] = percentage_match
        resume.replace_one(temp, d)
    return {"Percentage": final}


@app.route("/metrics", methods=["POST"])
def metrics():
    final = []
    data = request.json
    job_desc = data["job_desc"]
    data = resume.find()
    top, mid, low = 0, 0, 0
    best_100 = 0
    for d in data:
        temp = d.copy()
        resume_text = d["Resume_data"]
        content = [job_desc, resume_text]
        cv = CountVectorizer()
        count_matrix = cv.fit_transform(content)
        mat = cosine_similarity(count_matrix)
        percentage = round(mat[1][0] * 100)
        if percentage < 40:
            low += 1
        elif percentage >= 40 and percentage < 70:
            mid += 1
        else:
            top += 1
            if percentage == 100:
                best_100 += 1
        final.append(percentage)
        d["match_percentage"] = percentage
        resume.replace_one(temp, d)
    return {
        "Metrics": final,
        "percentage_table": [low, mid, top],
        "best_count": best_100,
    }


@app.route("/sortbymatch", methods=["GET"])
def sort_by():
    final = []
    data = resume.find().sort("match_percentage", -1)
    for d in data:
        try:
            del d["_id"]
        except:
            continue
        final.append(d)
    return {"data": final}


@app.route("/bestmatch", methods=["GET"])
def best_match():
    final = []
    data = resume.find().sort("match_percentage", -1)
    for d in data:
        try:
            del d["_id"]
        except:
            continue
        final.append(d)
        break
    return {"data": final}


@app.route("/topfive", methods=["GET"])
def top_five():
    final = []
    data = resume.find().sort("match_percentage", -1)
    count = 0
    for d in data:
        if count == 5:
            break
        try:
            del d["_id"]
        except:
            continue
        final.append(d)
        count += 1
    return {"data": final}


# Skill filtering API


@app.route("/skillfilter", methods=["POST"])
def skill_filter():
    inp = request.json
    skills_given = inp["skills_given"]
    final = []
    data = resume.find({"Skills": {"$all": skills_given}})
    for d in data:
        try:
            del d["_id"]
        except:
            continue
        final.append(d)
    return {"data": final}


@app.route("/analytics", methods=["GET"])
def analytics():
    low, mid, top, highest_percentage = 0, 0, 0, 0
    data = resume.find()
    temp_dicts = []
    for d in data:
        percentage = d["match_percentage"]
        if highest_percentage < percentage:
            highest_percentage = percentage
        skills = d["Skills"]
        temp_dicts.append(Counter(skills))
        try:
            del d["_id"]
        except:
            continue
        if percentage < 40:
            low += 1
        elif percentage >= 40 and percentage < 70:
            mid += 1
        else:
            top += 1
    final = Counter({})
    for t in temp_dicts:
        final += t
    final = dict(final)
    sorted_final = sorted(final.items(), key=lambda x: x[1], reverse=True)
    sorted_skills = [i[0] for i in sorted_final]
    try:
        sorted_skills = sorted_skills[:5]
    except:
        pass
    return {
        "percentage_table": [low, mid, top],
        "highest_percentage": highest_percentage,
        "top_skills": sorted_skills,
    }


if _name_ == "_main_":
    app.run(debug=True)