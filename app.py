import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# تحميل النموذج و Tokenizer
model_name = "aubmindlab/bert-base-arabertv02"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained("model")
# التصنيفات
labels = ["جبر", "تفاضل", "تكامل", "هندسة", "إحصاء"]
st.set_page_config(page_title="روان العتيبي | تصنيف مسائل الرياضيات")
st.title("تصنيف مسائل الرياضيات")
st.markdown("أدخل مسألة رياضية وسيقوم النموذج بتحديد نوعها (جبر، تفاضل، ...).")
# إدخال المستخدم
text = st.text_area("أدخل المسألة هنا:")
# زر التصنيف
if st.button("صنف"):
   if not text.strip():
       st.warning("يرجى إدخال مسألة.")
   else:
       inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
       with torch.no_grad():
           outputs = model(**inputs)
       pred = torch.argmax(outputs.logits, dim=1).item()
       st.success(f"التصنيف: **{labels[pred]}**")