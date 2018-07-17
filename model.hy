;; Translate Python -> Hy

(import time
        [numpy :as np]
        [tensorflow :as tf]
        config)

(defclass ChatBotModel [] []
  (defn --init-- [self forward-only batch-size]
    (print "initialize new model")
    (setv self.fw-only forward-only)
    (setv self.batch-size batch-size))
  (defn -create-placeholders [self]
    (print "Create placeholders")
    (setv self.encoder-inputs
          (list-comp (tf.placeholder tf.int32 :shape [None] :name (.format "encoder{}" i))
                     [i (range (. config.BUCKETS [-1] [0]))]))
    (setv self.decoder-inputs
          (list-comp (tf.placeholder tf.int32 :shape [None] :name (.format "decoder{}" i))
                     [i (range (inc (. config.BUCKETS [-1] [1])))]))
    (setv self.decoder-masks
          (list-comp (tf.placeholder tf.float32 :shape [None] :name (.format "mask{}" i))
                     [i (range (inc (. config.BUCKETS [-1] [1])))]))
    (setv self.targets (cut self.decoder-inputs 1)))
  (defn -inference [self]
    (print "Create inference")
    (if (and
          (< config.NUM-SAMPLES config.DEC-VOCAB)
          (> config.NUM-SAMPLES 0))
        (do
          (setv w (tf.get_variable "proj_w" [config.HIDDEN-SIZE config.DEC-VOCAB]))
          (setv b (tf.get_variable "proj_b" [config.DEC-VOCAB]))
          (setv self.output_projection (, w b))))
    (defn sampled-loss [logits labels]
      (setv labels (tf.reshape labels [-1 1]))
      (return
        (tf.nn.sampled_softmax_loss
          :weights (tf.transpose w)
          :biases b
          :inputs logits
          :labels labels
          :num-sampled config.NUM-SAMPLES
          :num-classes config.DEC-VOCAB)))
    (setv self.softmax-loss-function sampled-loss)
    (setv single-cell (tf.contrib.rnn.GRUCell config.HIDDEN-SIZE))
    (setv self.cell (tf.contrib.rnn.MultiRNNCell (list-comp single-cell [_ (range config.NUM-LAYERS)]))))
  (defn -create-loss [self] ;; what?
    (print "Creating loss ... \nIt might take a couple of minutes depending on how many buckets you have.")
    (setv start (.time time))
    (defn -seq2seq-f [encoder-inputs decoder-inputs do-decode]
      (setattr tf.contrib.rnn.GRUCell "__deepcopy__" (fn [self _] self))
      (setattr tf.contrib.rnn.MultiRNNCell "__deepcopy__" (fn [self _] self))
      (return
        (tf.contrib.legacy_seq2seq.embedding_attention_seq2seq
          encoder-inputs
          decoder-inputs
          self.cell
          :num-encoder-symbols config.ENC-VOCAB
          :num-decoder-symbols config.DEC-VOCAB
          :embedding-size config.HIDDEN-SIZE
          :output-projection self.output-projection
          :feed-previous do-decode)))
    (if self.fw-only ;; TODO : read more
        (do
          (setv (, self.outputs self.losses)
                ;; why self.- self.- = (tf.-) | hope self.- self.- = tf.-
                (tf.contrib.legacy_seq2seq.model_with_buckets
                  self.encoder-inputs
                  self.decoder-inputs
                  self.targets
                  self.decoder-masks
                  config.BUCKETS
                  (fn [x y] (-seq2seq-f x y True))
                  :softmax-loss-function self.softmax-loss-function))
          (if self.output_projection
              (for [bucket (range (len config.BUCKETS))]
                (setv (. self.outputs [bucket])
                      (list-comp
                        (+
                          (tf.matmul output (. self.output-projection [0]))
                          (. self.output_projection [1]))
                        [output (. self.outputs [bucket])])))))
        (setv (, self.outputs self.losses)
              (tf.contrib.legacy_seq2seq.model_with_buckets
                self.encoder-inputs
                self.decoder-inputs
                self.targets
                self.decoder-masks
                config.BUCKETS
                (fn [x y] (-seq2seq-f x y False))
                :softmax-loss-function self.softmax-loss-function))
        )
    (print "Time:" (- (.time time) start)))
  (defn -create-optimizer [self]
    (print "Create optimizer ... \nIt might take a couple of minutes depending on how many buckets you have.")
    (with [scope (tf.variable_scope "training")]
      (do
        (setv self.global-step
              (tf.Variable 0 :dtype tf.int32 :trainable False :name "global_step"))
        (if (not self.fw-only)
            (do
              (setv self.optimizer (tf.train.GradientDescentOptimizer config.LR))
              (setv trainables (tf.trainable_variables))
              (setv self.gradient-norms [])
              (setv self.train-ops [])
              (setv start (.time time))
              (for [bucket [range (len config.BUCKETS)]]
                (setv (, clipped-grads norm)
                      (tf.clip_by_global_norm
                        (tf.gradients (. self.losses [bucket]) trainables)
                        config.MAX-GRAD-NORM))
                (.append self.gradient-norms norm)
                (.append self.train-ops
                         (.apply_gradients self.optimizer (zip clipped-grads trainables)
                                           :global-step self.global-step))
                (print (.format "Creating opt for bucket {} took {} seconds"
                                bucket (- (.time time) start)))
                (setv start (.time time))))))))
  (defn -create-summary [self]
	)
  (defn build-graph [self]
    (self.-create-placeholders)
    (self.-inference)
    (self.-create-loss)
    (self.-create-optimizer)
    (self.-create-summary)))
