;; Translate Python -> Hylang

;; """ A neural chatbot using sequence to sequence model with
;; attentional decoder. 
;; This is based on Google Translate Tensorflow model 
;; https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/
;; Sequence to sequence model by Cho et al.(2014)
;; Created by Chip Huyen (chiphuyen@cs.stanford.edu)
;; CS20: "TensorFlow for Deep Learning Research"
;; cs20.stanford.edu
;; This file contains the hyperparameters for the model.
;; See README.md for instruction on how to run the starter code.
;;"""

;; parameters for processing the dataset
(setv DATA-PATH "./cornell movie-dialogs corpus")
(setv CONVO-FILE "movie_conversations.txt")
(setv LINE-FILE "movie_lines.txt")
(setv OUTPUT-FILE "output_convo.txt")
(setv PROCESSED-PATH "processed")
(setv CPT-PATH "checkpoints")

(setv THRESHOLD 2)
(setv PAD-ID 0)
(setv UNK-ID 1)
(setv START-ID 2)
(setv EOS-ID 3)

(setv TESTSET-SIZE 25000)

(setv BUCKETS [(, 19 19) (, 28 28) (, 33 33) (, 40 43) (, 50 53) (, 60 63)])
(setv CONTRACTIONS [(, "I ' m " "i ' m ") (, "' d " "'d ") (, "' s " "'s ")
                    (, "don ' t " "do n't ") (, "did n ' t " "did n't ")
                    (, "doesn ' t " "does n't ") (, "can ' t " "ca n't ")
                    (, "shouldn ' t " "should n't ") (, "wouldn ' t " "would n't ")
                    (, "' ve " "'ve ") (, "' re " "'re ")
                    (, "in ' " "in' ")])

(setv NUM-LAYERS 3)
(setv HIDDEN-SIZE 256)
(setv BATCH-SIZE 64)

(setv LR 0.5) ;; learning rate
(setv MAX_GRAD_NORM 5.0)

(setv NUM-SAMPLES 512)
(setv ENC-VOCAB 24295)
(setv DEC-VOCAB 24538)
