from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from flask import Flask, render_template, request,jsonify, redirect
import requests
import os
from PIL import Image
import pytesseract
from gtts import gTTS

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv( "HUGGINGFACEHUB_API_TOKEN")