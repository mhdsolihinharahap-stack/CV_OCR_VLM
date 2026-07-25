import os
import base64
import pandas as pd
import re
import io

from PIL import Image
from openai import OpenAI
from jiwer import cer
from tqdm import tqdm



client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)


MODEL = "Qwen3-VL-2B-Instruct-GGUF"



IMAGE_DIR = "dataset/Indonesian License Plate Dataset/images/test"

LABEL_DIR = "dataset/Indonesian License Plate Dataset/labelswithLP/test"

OUTPUT = "output/prediction.csv"



def encode_image(path):

    img = Image.open(path)

    # kecilkan gambar agar VLM lebih cepat
    img.thumbnail((1024,1024))


    buffer = io.BytesIO()


    img.save(
    buffer,
    format="JPEG",
    quality=95
)


    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")



def read_label(path):

    with open(path,"r",encoding="utf-8") as f:

        for line in f:

            data=line.strip().split()


            # labelswithLP format:
            # class x y w h PLATE

            if len(data) >= 6:

                return data[-1].upper()


    return ""



def normalize_plate(text):

    text=text.upper()

    text=re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text



def predict_plate(image_path):

    img64=encode_image(image_path)


    try:

        response=client.chat.completions.create(

            model=MODEL,


            messages=[

                {

                "role":"user",

                "content":[


                    {

                    "type":"text",

                    "text":
"""
You are an OCR system specialized in Indonesian vehicle license plates.

Read every visible character carefully.

Rules:
- Return ONLY the license plate.
- No spaces.
- No punctuation.
- No explanation.
- Use uppercase letters.
- Preserve every digit exactly.
- If a character is unclear, output the most likely character.

Example:
B1234ABC
"""

                    },


                    {

                    "type":"image_url",

                    "image_url":
                    {
                    "url":
                    f"data:image/jpeg;base64,{img64}"
                    }

                    }

                ]

                }

            ],


           temperature=0,
           top_p=0.1,
           frequency_penalty=0,
           presence_penalty=0,
           max_tokens=15,
           timeout=60
        )


        result = response.choices[0].message.content

        result = result.split("\n")[0]
        result = result.strip()

        return normalize_plate(result)



    except Exception as e:

        print(
            "OCR Error:",
            e
        )

        return ""


results=[]


images=sorted(

    [

    f for f in os.listdir(IMAGE_DIR)

    if f.lower().endswith(
        (".jpg",".jpeg",".png")
    )

    ]

)


print(
    f"Jumlah gambar: {len(images)}"
)



for img in tqdm(images):


    image_path=os.path.join(
        IMAGE_DIR,
        img
    )


    label_path=os.path.join(

        LABEL_DIR,

        os.path.splitext(img)[0]+".txt"

    )



    if not os.path.exists(label_path):

        print(
            "Label tidak ditemukan:",
            img
        )

        continue



    ground_truth=read_label(
        label_path
    )


    if ground_truth=="":

        print(
            "GT kosong:",
            img
        )

        continue



    prediction=predict_plate(
        image_path
    )


    score=cer(

        normalize_plate(ground_truth),

        prediction

    )



    results.append(

        {

        "image":img,

        "ground_truth":
        normalize_plate(ground_truth),

        "prediction":
        prediction,

        "CER_score":
        score

        }

    )



    print(

        f"{img} | "

        f"GT:{ground_truth} | "

        f"Pred:{prediction} | "

        f"CER:{score:.4f}"

    )



os.makedirs(
    "output",
    exist_ok=True
)



df=pd.DataFrame(results)



df.to_csv(
    OUTPUT,
    index=False
)



print("\nSelesai")

print(df)


print(
    "\nMean CER:",
    df["CER_score"].mean()
)