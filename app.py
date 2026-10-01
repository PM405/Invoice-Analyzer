import os
import base64

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai


load_dotenv()


app = Flask(__name__)


API_KEY = os.getenv("GOOGLE_API_KEY")


if not API_KEY:
    raise ValueError("GOOGLE_API_KEY is not configured.")


client = genai.Client(api_key=API_KEY)


ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def analyze_invoice(image_bytes, mime_type, question):

    image_data = base64.b64encode(
        image_bytes
    ).decode("utf-8")


    prompt = f"""
You are an expert AI invoice analysis assistant.

Carefully analyze the uploaded invoice.

Answer the user's question using only information
visible or readable in the invoice.

Do not invent information.

If the requested information is not available,
clearly say that it is not visible in the invoice.

You can identify:
- Invoice number
- Invoice date
- Seller name
- Buyer name
- Products
- Quantity
- Price
- Tax
- Discount
- Subtotal
- Total amount
- Payment information

User question:
{question}
"""


    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=[
            {
                "text": prompt
            },
            {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": image_data
                }
            }
        ]
    )


    return response.text


@app.route("/")
def home():

    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    try:

        if "invoice" not in request.files:

            return jsonify({
                "success": False,
                "message": "Please upload an invoice image."
            }), 400


        file = request.files["invoice"]


        if file.filename == "":

            return jsonify({
                "success": False,
                "message": "Please select an invoice image."
            }), 400


        if not allowed_file(file.filename):

            return jsonify({
                "success": False,
                "message": "Only JPG, JPEG, PNG and WEBP images are allowed."
            }), 400


        question = request.form.get(
            "question",
            ""
        ).strip()


        if not question:

            return jsonify({
                "success": False,
                "message": "Please enter a question."
            }), 400


        image_bytes = file.read()


        if not image_bytes:

            return jsonify({
                "success": False,
                "message": "The uploaded image is empty."
            }), 400


        mime_type = file.mimetype or "image/jpeg"


        result = analyze_invoice(
            image_bytes,
            mime_type,
            question
        )


        return jsonify({
            "success": True,
            "response": result
        })


    except Exception as error:

        print("Error:", error)


        return jsonify({
            "success": False,
            "message": "Unable to analyze the invoice.",
            "error": str(error)
        }), 500


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),
        debug=True
    )
