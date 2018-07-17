;; Translate Python -> Hylang

;; """ A neural chatbot using sequence to sequence model with
;; attentional decoder. 
;; This is based on Google Translate Tensorflow model 
;; https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/
;; Sequence to sequence model by Cho et al.(2014)
;; Created by Chip Huyen (chiphuyen@cs.stanford.edu)
;; CS20: "TensorFlow for Deep Learning Research"
;; cs20.stanford.edu
;; This file contains the code to do the pre-processing for the
;; Cornell Movie-Dialogs Corpus.
;; See readme.md for instruction on how to run the starter code.
;; """

(import config os random re [numpy :as np])

(defn get-lines []
  (setv id2line {})
  (setv file-path (os.path.join config.DATA-PATH config.LINE-FILE))
  (with [f (open file-path "r" :errors "ignore")]
    (setv i 0)
    (try
      (for [line f]
        (setv parts (.split line " +++$+++ "))
        (if (= 5 (len parts))
            (do
              (if (= "\n" (. parts [4] [-1]))
                  (setv (. parts [4]) (cut (. parts [4]) 0 -1)))
              (setv (. id2line [(. parts [0])]) (. parts [4]))))
        (setv i (inc i)))
      (except [e UnicodeDecodeError]
        (print i line))))
  (print i)
  (return id2line))

(defn get-convos []
  (setv file-path (os.path.join config.DATA-PATH config.CONVO_FILE))
  (setv convos [])
  (with [f (open file-path "r")]
    (for [line (.readlines f)]
      (setv parts (.split line " +++$+++ "))
      (if (= 4 (len parts))
          (do
            (setv convo [])
            (for [line (.split (cut (. parts [3]) 1 -2) ", ")]
              (.append convo (cut line 1 -1)))
            (.append convos convo)))))
  (return convos))

(defn question_answers [id2line convos]
  (setv questions [])
  (setv answers [])
  (for [convo convos]
    (for [[index line] (enumerate (cut convo 0 -1))]
      (do
        (.append questions (. id2line [(. convo [index])]))
        (.append answers (. id2line [(. convo [(inc index)])])))))
  (assert (= (len questions) (len answers)))
  (return (, questions answers)))

(defn prepare_dataset [questions answers]
  (make-dir config.PROCESSED_PATH)
  (setv test-ids (.sample random (list-comp i (i (range (len questions)))) config.TESTSET-SIZE))
  (setv filenames ["train.enc" "train.dec" "test.enc" "test.dec"])
  (setv files [])
  (for [filename filenames]
    (.append files (-> (os.path.join config.PROCESSED-PATH filename)
                       (open "w"))))
  (for [i (range (len questions))]
    (if (in i test_ids)
        (do
          (.write (. files [2]) (+ (. questions [i]) "\n"))
          (.write (. files [3]) (+ (. answers [i]) "\n")))
        (do
          (.write (. files [0]) (+ (. questions [i]) "\n"))
          (.write (. files [1]) (+ (. answers [i]) "\n")))))
  (for [file files]
    (.close file)))

(defn make-dir [path]
  (try
    (os.mkdir path)
    (except [e OSError]
      path)))

(defn basic_tokenizer [line &optional [normalize_digits True]]
  (setv line (re.sub  "<u>" "" line))
  (setv line (re.sub  "</u>" "" line))
  (setv line (re.sub  "\[" "" line))
  (setv line (re.sub  "\]" "" line))
  (setv words [])
  (setv -WORD-SPLIT (re.compile "([.,!?\"'-<>:;)(])"))
  (setv -DIGIT-RE (re.compile r"\d")) ;; TODO how to deal with escape sequence
  (for [fragment (-> (.strip line) (.lower) (.split))]
    (for [token (.split re -WORD-SPLIT fragment)]
      (if (not token)
          (continue))
      (if normalize_digits
          (setv token (.sub re -DIGIT-RE "#" token)))
      (.append words token)))
  (return words))

(defn build-vocab [filename &optional [normalize-digits True]]
  (setv in-path (os.path.join config.PROCESSED-PATH  filename))
  (setv out-path (os.path.join config.PROCESSED-PATH (.format "vocab.{}" (cut filename -3))))
  (setv vocab {})
  (with [f (open in-path "r")]
    (for [line (.readlines f)]
      (for [token (basic_tokenizer line)]
        (do
          (if (not (in token vocab))
              (setv (. vocab [token]) 0))
          (setv (. vocab [token]) (inc (. vocab [token])))))))
  (setv sorted-vocab (sorted vocab :key  vocab.get :reverse True))
  (with [f (open out-path "w")]
    (.write f (+ "<pad>""\n"))
    (.write f (+ "<unk>" "\n"))
    (.write f (+ "<s>" "\n"))
    (.write f (+ "<\s>" "\n"))
    (setv index 4)
    (for [word sorted-vocab]
      (do
        (when (< (. vocab [word]) config.THRESHOLD) ;; what is break if ?
          (break))
        (.write f (+ word "\n"))
        (setv index (inc index))))
    (with [cf (open "config.hy" "a")]
      (if (= (cut filename -3) "enc")
          (.write cf (+ "(setv " "ENC-VOCAB " (str index) ")" "\n"))
          (.write cf (+ "(setv " "DEC-VOCAB " (str index) ")" "\n"))))))

(defn load-vocab [vocab-path]
  (with [f (open vocab-path "r")]
    (setv words (-> (.read f) (.splitlines))))
  (return (, words (dict-comp (. words [i]) i [i (range (len words))]))))

(defn sentence2id [vocab line]
  (return (list-comp (.get vocab token (. vocab ["<unk>"])) [token (basic_tokenizer line)])))

(defn token2id [data mode]
  (setv vocab-path (+ "vocab." mode))
  (setv in-path (+ data "." mode))
  (setv out-path (+ data "_ids." mode))
  (setv (, _ vocab) (load-vocab (os.path.join config.PROCESSED-PATH vocab-path)))
  (setv in-file (open (os.path.join config.PROCESSED-PATH in-path) "r"))
  (setv out-file (open (os.path.join config.PROCESSED-PATH out-path) "w"))
  (setv lines (-> in-file (.read) (.splitlines)))
  (for [line lines]
    (do
      (if (= mode "dec")
          (setv ids [(. vocab ["<s>"])])
          (setv ids []))
      (.extend ids (sentence2id vocab line))
      (if (= mode "dec")
          (.append ids (. vocab ["<\s>"]))
          )
      (.write out-file (+ (.join " " (list-comp (str id_) [id_ ids])) "\n")))))

(defn prepare-raw-data []
  (print "Preparing raw data into train set and test set ...")
  (setv id2line (get-lines))
  (setv convos (get-convos))
  (setv (, questions answers) (question-answers id2line convos))
  (prepare-dataset questions answers))

(defn process-data []
  (print "Preparing data to be model-ready ...")
  (build-vocab "train.enc")
  (build-vocab "train.dec")
  (token2id "train" "enc")
  (token2id "train" "dec")
  (token2id "test" "enc")
  (token2id "test" "dec"))

(defn load-data [enc-filename dec-filename &optional [max-training-size None]]
  (setv encode-file (open (os.path.join config.PROCESSED-PATH enc-filename) "r"))
  (setv decode-file (open (os.path.join config.PROCESSED-PATH dec-filename) "r"))
  (setv encode (.readline encode-file));; ?
  ;; encode, decode = encode_file.readline(), decode_file.readline()
  (setv decode (.readline decode-file))
  (setv data-buckets (list-comp [] [_ config.BUCKETS]))
  (setv i 0)
  (while (and encode decode)
    (if (= 0 (% (+ i 1) 10000))
        (print "Bucketing conversation number" i)
        )
    (setv encode_ids (list-comp (int id_) [id_ (encode.split)]))
    (setv decode_ids (list-comp (int id_) [id_ (decode.split)]))
    (for [[bucket-id [encode-max-size decode-max-size]] (enumerate config.BUCKETS)]
      (if (and
            (<= (len decode-ids) decode-max-size)
            (<= (len encode-ids) encode-max-size))
          (do
            (.append (. data-buckets [bucket-id]) [encode-ids decode-ids])
            (break)))
      (setv encode (.readline encode-file))
      (setv decode (.readline decode-file))
      (setv i (inc i))))
  (return data-buckets))

(defn -pad-input (input_ size)
  (return (+ input_ (* [config.PAD_ID] (- size (len input_))))))

(defn -reshape-batch [inputs size batch-size]
  (setv batch-inputs [])
  (for [length-id (range size)]
    (.append batch-inputs (np.array (list-comp (. inputs [batch-id] [length-id])
                                               [batch-id (range batch-size)])  :dtype np.int32)))
  (return batch-inputs))

(defn get-batch [data-bucket bucket-id &optional [batch-size 1]]
  (setv (, encoder-size decoder-size) (. config.BUCKETS [bucket-id]))
  (setv encoder-inputs [])
  (setv decoder-inputs [])
  (for [_ (range batch-size)]
    (do
      (setv (, encoder-input decoder-input) (random.choice data-bucket))
      (.append encoder-inputs (list (reversed (-pad-input encoder-input encoder-size))))
      (.append decoder-inputs (-pad-input decoder-input decoder-size))))
  (setv batch-encoder-inputs (-reshape-batch encoder-inputs encoder-size batch-size))
  (setv batch-decoder-inputs (-reshape-batch decoder-inputs decoder-size batch-size))
  (setv batch-masks [])
  (for [length-id (range decoder-size)]
    (do
      (setv batch-mask (np.ones batch-size :dtype np.float32))
      (for [batch-id (range batch-size)]
        (do
          (if (< length-id (dec decoder-size))
              (setv target (. decoder-inputs [batch-id] [(inc length-id)])))
          (if (or
                (= target config.PAD_ID)
               (= length-id (dec decoder-size)))
              (setv (. batch-mask [batch-id]) 0.0))))
      (.append batch-masks batch-mask)))
  (return (, batch-encoder-inputs batch-decoder-inputs batch-masks)))

(if (= --name-- "__main__")
    (do
      (prepare-raw-data)
      (process-data)))
