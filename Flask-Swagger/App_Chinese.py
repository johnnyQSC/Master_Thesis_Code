##系統實作##
import flask
import torch
from flasgger import Swagger
from transformers import BertTokenizer
#建立應用程式
app=flask.Flask(__name__)
app.config['SWAGGER']={
  'title':'繁體中文事實查核系統',
  'uiversion':3
}
swagger=Swagger(app)
#載入BERT模型
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=torch.load('bert_model_chinese.pth', map_location=device)
model.eval()
#創建分詞器對象
tokenizer=BertTokenizer.from_pretrained('bert-base-chinese')
#定義API路徑
@app.route('/Traditional-Chinese-Fact-Checking',methods=['GET']) 
def ChineseFactChecking():
  """ (新聞資料庫更新日期:2022/10/31)
  ---
  parameters:
    - name: input_text
      in: query
      type: string
      required: true
      description: "請輸入繁體中文新聞標題文字資訊"
  responses:
    200:
      description: "本系統為您傳輸的資訊完成事實查核!"
  """  
  #取得使用者輸入
  input_text=flask.request.args.get('input_text')
  #將輸入的字串轉換為張量
  input_ids=torch.tensor(tokenizer.encode(input_text)).unsqueeze(0).to(device)
  #指定張量的資料型態
  input_ids=input_ids.to(torch.int64)
  #使用BERT模型進行預測
  with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
  probabilities=torch.softmax(logits,dim=-1).squeeze().tolist()
  #回傳預測結果
  return {"prediction":str(probabilities)}
if __name__=='__main__':
  app.run(debug='False',host='0.0.0.0',port=2023)