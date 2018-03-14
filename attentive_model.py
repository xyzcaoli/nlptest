def attentive_bilstm(input_data, input_label, wvs, is_training, num_pos=5, **kwargs):
    with tf.variable_scope('attentive_bilstm'):
        wv_params = tf.constant(wvs)[='']
        inputs = tf.nn.embedding_lookup(wv_params, input_data, None, name='inputs')
        state_size = kwargs.get('state_size', 100)
        lstm_num_layers = kwargs.get('lstm_num_layers', 2)
        cells_fw = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(state_size) for _ in range(lstm_num_layers)])
        cells_bw = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(state_size) for _ in range(lstm_num_layers)])
    
        initial_state_fw = cells_fw.zero_state(input_data.shape[0], tf.float32)
        initial_state_bw = cells_bw.zero_state(input_data.shape[0], tf.float32)
        
        rnn_inputs = [tf.squeeze(i,squeeze_dims=[1]) for i in tf.split(inputs, inputs.shape[1], axis= 1)]
        rnn_outputs, _, _ = tf.nn.static_bidirectional_rnn(cell_fw=cells_fw, cell_bw=cells_bw, inputs=rnn_inputs, 
                                                         initial_state_fw  = initial_state_fw, 
                                                           initial_state_bw = initial_state_bw)
       
        
        # 定义attention layer 

        attention_size = kwargs.get('attention_size', 100) #maybe need change
        sequence_length = kwargs.get('sequency_length',100) #need change
        with tf.name_scope('attention'), tf.variable_scope('attention'):
            attention_w = tf.Variable(tf.truncated_normal([2*state_size, attention_size], stddev=0.1), name='attention_w')
            attention_b = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b')
            u_list = []
            for t in xrange(sequence_length):
                u_t = tf.tanh(tf.matmul(rnn_outputs[t], attention_w) + attention_b) 
                u_list.append(u_t)
            u_w = tf.Variable(tf.truncated_normal([attention_size, 1], stddev=0.1), name='attention_uw')
            attn_z = []
            for t in xrange(sequence_length):
                z_t = tf.matmul(u_list[t], u_w)
                attn_z.append(z_t)
            # transform to batch_size * sequence_length
            attn_zconcat = tf.concat(attn_z, axis=1)
            self.alpha = tf.nn.softmax(attn_zconcat)
            # transform to sequence_length * batch_size * 1 , same rank as outputs
            alpha_trans = tf.reshape(tf.transpose(self.alpha, [1,0]), [sequence_length, -1, 1])
            self.final_output = tf.reduce_sum(rnn_outputs * alpha_trans, 0)
            
        outputs = tf.concat(final_output, axis=0, name='outputs')
        labels = tf.reshape(input_label, [-1])
        logits = tf.contrib.layers.fully_connected(outputs, num_pos, scope='logits')
        
        #prediction:
        prediction = tf.cast(tf.argmax(logits, 1), tf.int32)
        prediction = tf.reshape(prediction, [input_data.shape[0], input_data.shape[1]], name='prediction')
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = logits)
        loss = tf.reduce_mean(loss, name = 'loss')
        
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        correct = np.sum(top_k_op)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        
        return loss, prediction, accuracy

