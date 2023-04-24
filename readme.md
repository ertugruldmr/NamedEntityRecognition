<div style="position: absolute; top: 0; right: 0;">
    <a href="ertugrulbusiness@gmail.com"><img src="https://ssl.gstatic.com/ui/v1/icons/mail/rfr/gmail.ico" height="30"></a>
    <a href="https://tr.linkedin.com/in/ertu%C4%9Fruldemir?original_referer=https%3A%2F%2Fwww.google.com%2F"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" height="30"></a>
    <a href="https://github.com/ertugruldmr"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" height="30"></a>
    <a href="https://www.kaggle.com/erturuldemir"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/kaggle/kaggle-original.svg" height="30"></a>
    <a href="https://huggingface.co/ErtugrulDemir"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="30"></a>
    <a href="https://stackoverflow.com/users/21569249/ertu%c4%9frul-demir?tab=profile"><img src="https://upload.wikimedia.org/wikipedia/commons/e/ef/Stack_Overflow_icon.svg" height="30"></a>
    <a href="https://medium.com/@ertugrulbusiness"><img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Medium_icon.svg" height="30"></a>
    <a href="https://www.youtube.com/channel/UCB0_UTu-zbIsoRBHgpsrlsA"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/1024px-YouTube_full-color_icon_%282017%29.svg.png" height="30"></a>
</div>

# Named Entity Recognation
 
## __Table Of Content__
- (A) [__Brief__](#brief)
  - [__Project__](#project)
  - [__Data__](#data)
  - [__Demo__](#demo) -> [Live Demo](https://ertugruldemir-namedentityrecognition.hf.space)
  - [__Study__](#problemgoal-and-solving-approach) -> [Colab](https://colab.research.google.com/drive/1z002ybqI8M5ak0LAjHrD0mNeVAV4yzUB)
  - [__Results__](#results)
- (B) [__Detailed__](#Details)
  - [__Abstract__](#abstract)
  - [__Explanation of the study__](#explanation-of-the-study)
    - [__(A) Dependencies__](#a-dependencies)
    - [__(B) Dataset__](#b-dataset)
    - [__(C) Modelling__](#c-modelling)
    - [__(D) Deployment as web demo app__](#g-deployment-as-web-demo-app)
  - [__Licance__](#license)
  - [__Connection Links__](#connection-links)

## __Brief__ 

### __Project__ 
- This is a __Named Entitiy Recognition__ project on text data  with  __deep learning model__. The project uses the  [__Language-Independent Named Entity Recpgnition__](https://www.clips.uantwerpen.be/conll2003/ner/) to __Extract information__ from text.
- The __goal__ is build a deep learning model that accurately __extract informations__ as named entity from text data.
- The performance of the model is evaluated using several __metrics__ loss.

#### __Overview__
- This project involves building a deep learning model to extract structured information as named entity from text. Data files contain four columns separated by a single space. Each word has been put on a separate line and there is an empty line after each sentence. The first item on each line is a word, the second a part-of-speech (POS) tag, the third a syntactic chunk tag and the fourth the named entity tag. The models selected according to model tuning results, the progress optimized respectively the previous tune results. The project uses Python and several popular libraries such as Pandas, NumPy, tensorflow.

#### __Demo__

<div align="left">
  <table>
    <tr>
    <td>
        <a target="_blank" href="https://ertugruldemir-namedentityrecognition.hf.space" height="30"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="30">[Demo app] HF Space</a>
      </td>
      <td>
        <a target="_blank" href="https://colab.research.google.com/drive/1f-_tk0nMSXod8n8jr9ch1PGuPw2HJfO-"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">[Demo app] Run in Colab</a>
      </td>
      <td>
        <a target="_blank" href="https://github.com/ertugruldmr/NamedEntityRecognition/blob/main/study.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png">[Traning pipeline] source on GitHub</a>
      </td>
    <td>
        <a target="_blank" href="https://colab.research.google.com/drive/1z002ybqI8M5ak0LAjHrD0mNeVAV4yzUB"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">[Traning pipeline] Run in Colab</a>
      </td>
    </tr>
  </table>
</div>


- Description
    -  __Extract structured information__ from text data.
    - __Usage__: 
      - Write your text then clict the button for extraction
- Embedded [Demo](https://ertugruldemir-namedentityrecognition.hf.space) window from HuggingFace Space
    

<iframe
	src="https://ertugruldemir-namedentityrecognition.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

#### __Data__
- The [__Language-Independent Named Entity Recpgnition__](https://www.clips.uantwerpen.be/conll2003/ner/) from huggingface dataset api.
- Data files contain four columns separated by a single space. 
  - Each word has been put on a separate line and there is an empty line after each sentence. 
  - The first item on each line is a word, the second a part-of-speech (POS) tag, the third a syntactic chunk tag and the fourth the named entity tag.

#### Problem, Goal and Solving approach
- TThis is a __Named Entitiy Recognition__ problem  that uses the  [__Language-Independent Named Entity Recpgnition__](https://www.clips.uantwerpen.be/conll2003/ner/)  to __Extract information__ from given image.
- The __goal__ is build a deep learning  model that accurately __extract informations__ as named entity from text data.
- __Solving approach__ is that using the supervised deep learning models. Basic Custom convolutional model is used for information extraction. 

#### Study
The project aimed predicting the age and gender using deep learning model architecture. The study includes following chapters.
- __(A) Dependencies__: Installations and imports of the libraries.
- __(B) Dataset__: Downloading and loading the dataset. Preparing the dataset from official website. Configurating the dataset performance and related pre-processes. 
- __(C) Preprocessing__: normalizing the text data, splitting the text data, tokenization, padding, vectorization, configurating the dataset object, batching, performance setting, visualizating, Implementing related processing methods on train dataset and text related processes.
- __(D) Modelling__:
  - Model Architecture
    - Custom convolutional neural network model used. The model includes transformer and multi head attention mechanism.
  - Training
    - Callbakcs and trainin params are setted. some of the callbacks are EarlyStopping, ModelCheckpoint, Tensorboard etc....  
    - TTrained the model on the text data over 10 epochs    
  - Saving the model
    - Saved the model as tensorflow saved model format.
- __(E) Deployment as web demo app__: Creating Gradio Web app to Demostrate the project.Then Serving the demo via huggingface as live.

#### results
- The final model is __Custom Transformer Multi Head Attention Network__ as Custom tranformer network which includes attention mechanicsm because of the results and less complexity.
  -  Custom Transformer Multi Head Attention Network
        <table><tr><th>Model Results </th><th></th></tr><tr><td>
  |   | loss  |
  |---|-------|
  |   | loss: 0.0424|
    </td></tr></table>

## Details

### Abstract
- [__Language-Independent Named Entity Recpgnition__](https://www.clips.uantwerpen.be/conll2003/ner/)  is used to extract structural informations from text data. The dataset contains four columns separated by a single space. Each word has been put on a separate line and there is an empty line after each sentence. The first item on each line is a word, the second a part-of-speech (POS) tag, the third a syntactic chunk tag and the fourth the named entity tag. The goal is build a deep learning model that accurately extract informations as named entity from text data using through custom deep learning algorithms which include attention mechanicsm as transfomer or related training approachs of pretrained state of art models.The study includes creating the environment, getting the data, preprocessing the data, exploring the data, mormalizing the text data, splitting the text data, tokenization, padding, vectorization, configurating the dataset object, batching, performance setting, visualizating, modelling the data, saving the results, deployment as demo app. Training phase of the models implemented through tensorflow callbacks. After the custom model traininigs, transfer learning and fine tuning approaches are implemented. Selected the basic and more succesful when comparet between other models  is custom  tranformer model which include attention mechanicsm. __Custom Transformer Multi Head Attention Network__ model  has __0.0424__ loss,  other metrics are also found the results section. Created a demo at the demo app section and served on huggingface space.  


### File Structures

- File Structure Tree
```bash
├── demo_app
│   ├── app.py
│   ├── ner_model
│   └── vocabulary
├── docs
│   └── images
├── env
│   └── env_installation.md
│   └── requirements.txt
├── LICENSE
├── readme.md
└── study.ipynb
```
- Description of the files
  - demo_app/
    - Includes the demo web app files, it has the all the requirements in the folder so it can serve on anywhere.
  - demo_app/ner_model:
    - Custom transformer Model Which includes attention mechanism and saved as tensorflow saved_model format.
  - docs/
    - Includes the documents about results and presentations
  - env/
    - It includes the training environmet related files. these are required when you run the study.ipynb file.
  - LICENSE.txt
    - It is the pure apache 2.0 licence. It isn't edited.
  - readme.md
    - It includes all the explanations about the project
  - study.ipynb
    - It is all the studies about solving the problem which reason of the dataset existance. 
  - requirements.txt
    - It includes the library dependencies of the study.   

### Explanation of the Study
#### __(A) Dependencies__:
  - You can create an environment via env/requirements.txt. Create a virtual environment then use hte following code. It is enough to satisfy the requirements for runing the study.ipynb which training pipeline.
  - Dataset can download from huggingface dataset.
#### __(B) Dataset__: 
  - Downloading the [__Language-Independent Named Entity Recpgnition__](https://www.clips.uantwerpen.be/conll2003/ner/)  via huggingface dataset api. 
  -  Each word has been put on a separate line and there is an empty line after each sentence.
    - The first item on each line is a word,
    - the second a part-of-speech (POS) tag, 
    - the third a syntactic chunk tag  
    - the fourth the named entity tag. 
  - Preparing the dataset via resizing, mormalizing the text data, splitting the text data, tokenization, padding, vectorization, configurating the dataset object, text preprocessing processes. 




#### __(C) Modelling__: 
  - The processes are below:
    - Model Architecture
      - Custom convolutional neural network model used. The model includes transformer and multi head attention mechanism.
  - Training
      - Custom Convolutional Model Archirecture
    - Training
      - Callbakcs and trainin params are setted. some of the callbacks are EarlyStopping, ModelCheckpoint, Tensorboard etc....  
      - Trained the model on the text data over 10 epochs 
    - Saving the model
      - Saved the model as tensorflow saved model format.
  - __(E) Deployment as web demo app__: Creating Gradio Web app to Demostrate the project.Then Serving the demo via huggingface as live.

  #### results
  - The final model is __Custom Transformer Multi Head Attention Network__ as Custom tranformer network which includes attention mechanicsm because of the results and less complexity.
  -  Custom Transformer Multi Head Attention Network
        <table><tr><th>Model Results </th><th></th></tr><tr><td>
  |   | loss  |
  |---|-------|
  |   | loss: 0.0424|
  </td></tr></table>
    - Saving the project and demo studies.
      - trained model __Custom Transformer Multi Head Attention Network__ as tensorflow (keras) saved_model format.

#### __(D) Deployment as web demo app__: 
  - Creating Gradio Web app to Demostrate the project.Then Serving the demo via huggingface as live.
  - Desciption
    - Project goal is predicting age and gender of any person from given image.
    - Usage: write your text for information extraction then use the button to predict.
  - Demo
    - The demo app in the demo_app folder as an individual project. All the requirements and dependencies are in there. You can run it anywhere if you install the requirements.txt.
    - You can find the live demo as huggingface space in this [demo link](https://ertugruldemir-namedentityrecognition.hf.space) as full web page or you can also us the [embedded demo widget](#demo)  in this document.  
    
## License
- This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

<h1 style="text-align: center;">Connection Links</h1>

<div style="text-align: center;">
    <a href="ertugrulbusiness@gmail.com"><img src="https://ssl.gstatic.com/ui/v1/icons/mail/rfr/gmail.ico" height="30"></a>
    <a href="https://tr.linkedin.com/in/ertu%C4%9Fruldemir?original_referer=https%3A%2F%2Fwww.google.com%2F"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" height="30"></a>
    <a href="https://github.com/ertugruldmr"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" height="30"></a>
    <a href="https://www.kaggle.com/erturuldemir"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/kaggle/kaggle-original.svg" height="30"></a>
    <a href="https://huggingface.co/ErtugrulDemir"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="30"></a>
    <a href="https://stackoverflow.com/users/21569249/ertu%c4%9frul-demir?tab=profile"><img src="https://upload.wikimedia.org/wikipedia/commons/e/ef/Stack_Overflow_icon.svg" height="30"></a>
    <a href="https://www.hackerrank.com/ertugrulbusiness"><img src="https://hrcdn.net/fcore/assets/work/header/hackerrank_logo-21e2867566.svg" height="30"></a>
    <a href="https://app.patika.dev/ertugruldmr"><img src="https://app.patika.dev/staticFiles/newPatikaLogo.svg" height="30"></a>
    <a href="https://medium.com/@ertugrulbusiness"><img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Medium_icon.svg" height="30"></a>
    <a href="https://www.youtube.com/channel/UCB0_UTu-zbIsoRBHgpsrlsA"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/1024px-YouTube_full-color_icon_%282017%29.svg.png" height="30"></a>
</div>

