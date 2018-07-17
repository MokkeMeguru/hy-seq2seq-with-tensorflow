;; Translate Python -> Hy

;; """ A neural chatbot using sequence to sequence model with
;; attentional decoder. 
;; This is based on Google Translate Tensorflow model 
;; https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/
;; Sequence to sequence model by Cho et al.(2014)
;; Created by Chip Huyen (chiphuyen@cs.stanford.edu)
;; CS20: "TensorFlow for Deep Learning Research"
;; cs20.stanford.edu
;; This file contains the code to run the model.
;; See README.md for instruction on how to run the starter code.
;; """

(import argparse
        os)
(setv (. os.environ ["TF_CPP_MIN_LOG_LEVEL"]) (str 2))
(import random
        sys
        time
        [numpy :as np]
        [tensorflow :as tf]
        [model [ChatBotModel]]
        config
        data)

(defn -get-random-bucket [train-buckets-scale]
  (setv rand (.random ramdom))
  (return (min (list-comp i [i (range (len train-buckets-scale))]
                          (if (> (. train-buckets-scale [i]) rand))))))

(defn -assert-lengths [encoder-size decoder-size encoder-inputs decoder-inputs decoder-masks]
  (when (!= (len encoder-inputs) encoder-size)
      (raise (ValueError
               (.format "Encode length must be equal to the one in bucketm, {} != {}."
                        (len encoder-inputs) encoder-size))))
  (when (!= (len decoder-inputs) decoder-size)
    (raise (ValueError
             (.format "Decode length must be equal to the one in bucketm, {} != {}."
                      (len decoder-inputs) decoder-size))))
  (when (!= (len decoder-masks) decoder-size)
    (raise (ValueError
             (.format "Weights length must be equal to the one in bucketm, {} != {}."
                      (len decoder-masks) decoder-size)))))

(defn run-step [sess model encoder-inputs decoder-inputs decoder-masks bucket-id forward-only]
  (setv (, encoder-size decoder-size) (. config.BUCKETS [bucket-id]))
  (-assert-lengths encoder-size decoder-size encoder-inputs decoder-inputs decoder-masks)
  (setv input-feed {})
  (for [step (range encoder-size)]
    (setv (. input-feed [(. (. model.encoder-inputs [step]) name)]) (. encoder-inputs [step])))
  (for [step (range decoder-size)]
    (do
      (setv (. input-feed [(. (. model.decoder-inputs [step]) name)])
            (. decoder-inputs [step]))
      (print (type decoder-masks))
      (setv (. input-feed [(. (. model.decoder-masks [step]) name)])
            (. decoder-masks [step]))))
  (setv last-target (. (. model.decoder-inputs [decoder-size]) name))
  (setv (. input-feed [last-target]) (np.zeros [model.batch-size] :dtype np.int32))
  (if (not forward-only)
      (setv output-feed [(. model.train-ops [bucket-id])
                         (. model.gradient-norms [bucket-id])
                         (. model.losses [bucket-id])])
      (do
        (setv output-feed [(. model.losses [bucket-id])])
        (for [step (range decoder-size)]
          (.append output-feed (. model.outputs [bucket-id] [step])))))
  (setv outputs (.run sess output-feed input-feed))
  (if (not forward-only)
      (return (, (. outputs [1]) (. outputs [2]) None))  ;; gradiend-norm loss output(none)
      (return (, None (. outputs [0]) (cut outputs 1)))))

(defn -get-buckets []
  (setv test-buckets (data.load-data "test_ids.enc" "test_ids.dec"))
  (setv data-buckets (data.load-data "train_ids.enc" "train_ids.dec"))
  (setv train-bucket-sizes (list-comp (len (. data-buckets [b]))
                                      [b range (len config.BUCKETS)]))
  (print "Number of samples in each bucket:\n" train-bucket-sizes)
  (setv train-total-size (sum train-bucket-sizes))
  (setv train-buckets-scale
        (list-comp (/ (sum (cut train-bucket-sizes (inc i)))
                      train-total-size)
                   [i (range (len train-bucket-sizes))]))
  (print "Buckat scale:\n" train-buckets-scale)
  (return (, test-buckets data-buckets train-buckets-scale)))

(defn -get-skip-step [iteration]
  (if (< iteration 100)
      (return 30)
      (return 100)))

(defn -check-restore-parameters [sess saver]
  (setv ckpt (tf.train.get_checkpoint_state
               (os.path.dirname (+ config.CPT_PATH "/checkpoint"))))
  (if (and ckpt
           ckpt.model_checkpoint_path)
      (do
        (print "Loading parameters for the Chatbot")
        (saver.restore sess ckpt.model_checkpoint_path))
      (print "Initializing fresh parameters for the Chatbot")))

(defn -eval-test-set [sess model test-buckets]
  (for [bucket-id (range (len config.BUCKETS))]
    (when (= (len (. test-buckets [bucket-id])) 0)
        (do
          (print (.format " Test: empty bucket {}" bucket-id))
          (continue)))
    (setv start (.time time))
    (setv (, encoder-inputs decoder-inputs decoder-masks)
          (data.get-batch
            (. test-buckets [bucket-id])
            bucket-id
            :batch-size config.BATCH_SIZE))
    (setv (, _ step-loss _) (run-step
                              sess model encoder-inputs decoder-inputs decoder-masks
                              bucket-id True))
    (print (.format "Test bucket {}: loss {}, time {}"
                    bucket-id step-loss (- (.time time) start)))))

(defn train []
  (setv (, test-buckets data-buckets train-buckets-scale) (-get-buckets))
  (setv model (ChatBotModel False config.BATCH_SIZE))
  (.build-graph model)
  (setv saver (tf.train.Saver))
  (with [sess (tf.Session)]
    (print "Running session")
    (.run sess (tf.global_variables_initializer))
    (-check-restore-parameters sess saver)
    (setv iteration (.eval model.global_step))
    (setv total-loss 0)
    (while True
      (do
        (setv skip-step (-get-skip-step iteration))
        (setv bucket-id (-get-random-bucket train-buckets-scale))
        (setv (, encoder-inputs decoder-inputs decoder-masks)
              (.get-batch data
                          (. data-buckets [bucket-id])
                          bucket-id
                          :batch-size config.BATCH_SIZE))
        (setv start (.time time))
        (setv (, _ step-loss _)
              (run-step sess model
                        encoder-inputs decoder-inputs decoder-masks bucket False))
        (setv total-loss (+ step-loss total-loss))
        (setv iteration (inc iteration))
        (when (= 0 (% iteration skip-step))
            (do
              (print (.format "Iter {}: loss {}, time {}"
                              iteration (/ total-loss skip-step) (- (.time time) start)))
              (setv start (.time time))
              (setv total-loss 0)
              (.save saver
                     sess (os.path.join config.CPT_PATH "chatbot")
                     :global-step (. model global_step))
              (when (= 0 (% iteration (* 10 skip-step)))
                  (do
                    (-eval-test-set sess model test-buckets)
                    (setv start (.time time))))
              (.flush sys.stdout)))))))

(defn -get-user-input []
  (print "> " :end "")
  (.flush sys.stdout)
  (return (.readline sys.stdin)))

(defn -find-right-bucket [length]
  (return
    (min (list-comp
           b [b (range (len config.BUCKETS))]
           (if (>= (. config.BUCKETS [b] [0]) length))))))

(defn -construct-response [output-logits inv-dec-vocab]
  (print (. output-logits [0]))
  (setv outputs (list-comp (int (np.argmax logit :axis 1)) [logit output-logits]))
  (if (in config.EOS_ID outputs)
      (setv outputs (cut outputs (.index outputs config.EOS_ID))))
  (return (.join " " (list-comp (tf.compat.as_str (. inv-dec-vocab [output]))
                                [output outputs]))))

(defn chat []
  (setv (, _ enc-vocab) (.load-vocab data (os.path.join config.PROCESSED-PATH "vocab.enc")))
  (setv (, inv-dec-vocab _) (.load-vocab data (os.path.join config.PROCESSED-PATH "vocab.dec")))
  (setv model (ChatBotModel True :batch-size 1))
  (.build-graph model)
  (setv saver (tf.train.Saver))
  (with [sess (tf.Session)]
    (.run sess (tf.global_variables_initializer))
    (-check-restore-parameters sess saver)
    (setv output-file (open (os.path.join config.PROCESSED-PATH config.OUTPUT-FILE) "a+"))
    (setv max-length (. config.BUCKETS [-1] [0]))
    (print "Welcome to TensorBro. Say something. Enter to exit. Max length is" max-length)
    (while True
      (do
        (setv line (-get-user-input))
        (when (and
                (= "\n" (. line [-1]))
                (> (len line) 0))
          (setv line (cut line 0 -1)))
        (when (= line "")
          (break))
        (.write output-file (+ "HUMAN ++++ " line "\n"))
        (setv token-ids (.sentence2id data enc-vocab (str line)))
        (when (> (len token-ids) max-length)
          (do
            (print "Max length I can handle is" max-length)
            (setv line (-get-user-input))
            (continue)))
        (setv bucket-id (-find-right-bucket (len token-ids)))
        (setv (, encoder-inputs decoder-inputs decoder-masks)
              (.get-batch data
                          [(, token-ids [])]
                          bucket-id
                          :batch-size 1))
        (setv (, _ _ output-logits)
              (run-step sess model encoder-inputs decoder-inputs
                        decoder-masks bucket-id True))
        (setv response (-construct-response output-logits inv-dec-vocab))
        (print response)
        (.write output-file (+ "BOT ++++ " response "\n"))))
    (.write output-file "=========================================\n")
    (.close output-file)))

(defn main []
  (setv parser (.ArgumentParser argparse))
  (parser.add_argument "--mode" :choices ["train" "chat"]
                       :default "train"
                       :help "mode. if not specified it's in the train mode")
  (setv args (.parse_args parser))
  (if (not (os.path.join config.PROCESSED-PATH))
      (do
        (data.prepare-raw-data)
        (data.process-data)))
  (print "Data ready!")
  (.make-dir data config.CPT_PATH)
  (if (= "train"(. args mode))
      (do
        (print "train-mode")
        (train)))
  (when (= "chat" (. args mode))
      (do
        (print "chat-mode")
        (chat))))

(if (= --name-- "__main__")
    (main))
