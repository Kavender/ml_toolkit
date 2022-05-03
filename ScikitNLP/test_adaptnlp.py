from adaptnlp import EasyTokenTagger
from adaptnlp import EasySequenceClassifier
from adaptnlp import EasyQuestionAnswering



if __name__ == "__main__":
    ## Example Text
    example_text = "Novetta is a great company that was chosen as one of top 50 great places to work! Novetta's headquarters is located in Mclean, Virginia."

    ## Load the token tagger module and tag text with the NER model
    tagger = EasyTokenTagger()
    sentences = tagger.tag_text(text=example_text, model_name_or_path="ner")
    ## Output tagged token span results in Flair's Sentence object model
    for sentence in sentences:
        for entity in sentence.get_spans("ner"):
            print(entity)
    print("-------------------------------")

    ## Load the sequence classifier module and classify sequence of text with the english sentiment model
    classifier = EasySequenceClassifier()
    sentences = classifier.tag_text(text=example_text, mini_batch_size=1, model_name_or_path="en-sentiment")
    ## Output labeled text results in Flair's Sentence object model
    for sentence in sentences:
        print(sentence.labels)
    print("-------------------------------")

    query = "What is the meaning of life?"
    context = "Machine Learning is the meaning of life."
    top_n = 5
    ## Load the QA module and run inference on results
    qa = EasyQuestionAnswering()
    best_answer, best_n_answers = qa.predict_qa(query=query, context=context, n_best_size=top_n, mini_batch_size=1, model_name_or_path="distilbert-base-uncased-distilled-squad")
    ## Output top answer as well as top 5 answers
    print("best_answer: ", best_answer)
    print("N best answers:", best_n_answers)
    print("-------------------------------")
