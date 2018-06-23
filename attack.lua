require 'nn'
require 'nngraph'
require 'hdf5'

require 's2sa.data'
require 's2sa.models'
require 's2sa.model_utils'
require 'util.char_manipulations'
require 'util.string_functions'
require 'util.beam_search'
require 'util.lang'
cmd = torch.CmdLine()

-- data files
cmd:text("")
cmd:text("**Data options**")
cmd:text("")
--cmd:option('-black',0,[[]])
cmd:option('-likely',2,[[nth likeliest word]])
cmd:option('-controlled',0,[[this script perforrms targeted attacks in the paper]])
cmd:option('-data_file','data/demo-train.hdf5', [[Path to the training *.hdf5 file from preprocess.py]])
cmd:option('-val_data_file','data/demo-val.hdf5', [[Path to validation *.hdf5 file from preprocess.py]])
cmd:option('-test_data_file','data/demo-test.hdf5', [[Path to validation *.hdf5 file from preprocess.py]])
cmd:option('-savefile', 'seq2seq_lstm_attn', [[Savefile name (model will be saved as
                                             savefile_epochX_PPL.t7 where X is the X-th epoch and PPL is
                                             the validation perplexity]])
cmd:option('-num_shards', 0, [[If the training data has been broken up into different shards,
                             then training files are in this many partitions]])
cmd:option('-saved_model', '', [[If training from a checkpoint then this is the path to the pretrained model.]])
cmd:option('-beam', 1, [[use greedy decoding for translation]])
cmd:option('-beam_size', 5 , [[adversary beam size]])
cmd:option('-targ_dict', '', [[targ_dict]])
cmd:option('-src_dict', '', [[src_dict]])
cmd:option('-char_dict', '', [[char_dict]])
cmd:option('-perc_',0.2, [[percentage that the adversary manipulates characters]])
-- rnn model specs
cmd:text("")
cmd:text("**Model options**")
cmd:text("")
cmd:text("**Model options**")
cmd:text("")

cmd:option('-num_layers', 2, [[Number of layers in the LSTM encoder/decoder]])
cmd:option('-rnn_size', 500, [[Size of LSTM hidden states]])
cmd:option('-word_vec_size', 300, [[Word embedding sizes]])
cmd:option('-attn', 1, [[If = 1, use attention on the decoder side. If = 0, it uses the last
                       hidden state of the decoder as context at each time step.]])
cmd:option('-brnn', 0, [[NO SUPOORT FOR BRNN]])
cmd:option('-use_chars_enc', 1, [[If = 1, use character on the encoder side (instead of word embeddings]])
cmd:option('-use_chars_dec', 0, [[If = 1, use character on the decoder side (instead of word embeddings]])
cmd:option('-reverse_src', 0, [[If = 1, reverse the source sequence. The original
                              sequence-to-sequence paper found that this was crucial to
                              achieving good performance, but with attention models this
                              does not seem necessary. Recommend leaving it to 0]])
cmd:option('-init_dec', 1, [[Initialize the hidden/cell state of the decoder at time
                           0 to be the last hidden/cell state of the encoder. If 0,
                           the initial states of the decoder are set to zero vectors]])
cmd:option('-input_feed', 1, [[If = 1, feed the context vector at each time step as additional
                             input (vica concatenation with the word embeddings) to the decoder]])
cmd:option('-multi_attn', 0, [[If > 0, then use a another attention layer on this layer of
                             the decoder. For example, if num_layers = 3 and `multi_attn = 2`,
                             then the model will do an attention over the source sequence
                             on the second layer (and use that as input to the third layer) and
                             the penultimate layer]])
cmd:option('-res_net', 0, [[Use residual connections between LSTM stacks whereby the input to
                          the l-th LSTM layer if the hidden state of the l-1-th LSTM layer
                          added with the l-2th LSTM layer. We didn't find this to help in our
                          experiments]])
cmd:option('-guided_alignment', 0, [[If 1, use external alignments to guide the attention weights as in
                                   (Chen et al., Guided Alignment Training for Topic-Aware Neural Machine Translation,
                                   arXiv 2016.). Alignments should have been provided during preprocess]])
cmd:option('-guided_alignment_weight', 0.5, [[default weights for external alignments]])
cmd:option('-guided_alignment_decay', 1, [[decay rate per epoch for alignment weight - typical with 0.9,
                                         weight will end up at ~30% of its initial value]])
cmd:text("")
cmd:text("Below options only apply if using the character model.")
cmd:text("")

cmd:option('-char_vec_size', 25, [[Size of the character embeddings]])

cmd:option('-kernel_width', {1,2,3,4,5,6,7}, [[Size (i.e. width) of the convolutional filter]])
cmd:option('-num_kernels', {50,100,150,200,200,200,200}, [[Number of convolutional filters (feature maps). So the
                                 representation from characters will have this many dimensions]])

cmd:option('-num_highway_layers', 2, [[Number of highway layers in the character model]])

cmd:text("")
cmd:text("**Optimization options**")
cmd:text("")

cmd:option('-epochs', 20, [[Number of training epochs]])
cmd:option('-start_epoch', 1, [[If loading from a checkpoint, the epoch from which to start]])
cmd:option('-param_init', 0.1, [[Parameters are initialized over uniform distribution with support (-param_init, param_init)]])
cmd:option('-optim', 'sgd', [[Optimization method. Possible options are: sgd (vanilla SGD), adagrad, adadelta, adam]])
cmd:option('-learning_rate', 1, [[Starting learning rate. If adagrad/adadelta/adam is used,
                                then this is the global learning rate. Recommended settings: sgd =1,
                                adagrad = 0.1, adadelta = 1, adam = 0.1]])
cmd:option('-max_grad_norm', 5, [[If the norm of the gradient vector exceeds this renormalize it to have the norm equal to max_grad_norm]])
cmd:option('-dropout', 0.3, [[Dropout probability. Dropout is applied between vertical LSTM stacks.]])
cmd:option('-emdropout', 0, [[Dropout probability on embeddings]])
cmd:option('-lr_decay', 0.5, [[Decay learning rate by this much if (i) perplexity does not decrease
                             on the validation set or (ii) epoch has gone past the start_decay_at_limit]])
cmd:option('-start_decay_at', 9, [[Start decay after this epoch]])
cmd:option('-curriculum', 0, [[For this many epochs, order the minibatches based on source
                             sequence length. Sometimes setting this to 1 will increase convergence speed.]])
cmd:option('-feature_embeddings_dim_exponent', 0.7, [[If the feature takes N values, then the
                                                    embbeding dimension will be set to N^exponent]])
cmd:option('-pre_word_vecs_enc', '', [[If a valid path is specified, then this will load
                                     pretrained word embeddings (hdf5 file) on the encoder side.
                                     See README for specific formatting instructions.]])
cmd:option('-pre_word_vecs_dec', '', [[If a valid path is specified, then this will load
                                     pretrained word embeddings (hdf5 file) on the decoder side.
                                     See README for specific formatting instructions.]])
cmd:option('-fix_word_vecs_enc', 0, [[If = 1, fix word embeddings on the encoder side]])
cmd:option('-fix_word_vecs_dec', 0, [[If = 1, fix word embeddings on the decoder side]])
cmd:option('-max_batch_l', '', [[If blank, then it will infer the max batch size from validation
                               data. You should only use this if your validation set uses a different
                               batch size in the preprocessing step]])

cmd:text("")
cmd:text("**Other options**")
cmd:text("")

cmd:option('-start_symbol', 0, [[Use special start-of-sentence and end-of-sentence tokens
                               on the source side. We've found this to make minimal difference]])
cmd:option('-gpuid', 1, [[Which gpu to use. -1 = use CPU]])
cmd:option('-gpuid2', -1, [[If this is >= 0, then the model will use two GPUs whereby the encoder
                          is on the first GPU and the decoder is on the second GPU.
                          This will allow you to train with bigger batches/models.]])
cmd:option('-cudnn', 1, [[Whether to use cudnn or not for convolutions (for the character model).
                        cudnn has much faster convolutions so this is highly recommended
                        if using the character model]])
cmd:option('-save_every', 10, [[Save every this many epochs]])
cmd:option('-print_every', 200, [[Print stats after this many batches]])
cmd:option('-seed', 3435, [[Seed for random initialization]])
cmd:option('-prealloc', 1, [[Use memory preallocation and sharing between cloned encoder/decoders]])


function zero_table(t)
  for i = 1, #t do
    if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
      if i == 1 then
        cutorch.setDevice(opt.gpuid)
      else
        cutorch.setDevice(opt.gpuid2)
      end
    end
    t[i]:zero()
  end
end

function append_table(dst, src)
  for i = 1, #src do
    table.insert(dst, src[i])
  end
end

function stop_words(file)
   local f = io.open(file,'r')
    local t = {}
    for line in f:lines() do
      local c = {}
      for w in line:gmatch'([^%s]+)' do
        table.insert(c, w)
      end
      t[c[1]] = 1
    end
    return t
end

function generate_adv(data)
  local num_params = 0
  local num_prunedparams = 0
  params, grad_params = {}, {}
  for i = 1, #layers do
    if opt.gpuid2 >= 0 then
      if i == 1 then
        cutorch.setDevice(opt.gpuid)
      else
        cutorch.setDevice(opt.gpuid2)
      end
    end
    local p, gp = layers[i]:getParameters()
    num_params = num_params + p:size(1)
    params[i] = p
    grad_params[i] = gp
    layers[i]:apply(function (m) if m.nPruned then num_prunedparams=num_prunedparams+m:nPruned() end end)
  end

  print("Number of parameters: " .. num_params .. " (active: " .. num_params-num_prunedparams .. ")")

  if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
    cutorch.setDevice(opt.gpuid)
    word_vec_layers[1].weight[1]:zero()
    cutorch.setDevice(opt.gpuid2)
    word_vec_layers[2].weight[1]:zero()
  else
    word_vec_layers[1].weight[1]:zero()
    word_vec_layers[2].weight[1]:zero()
  end

  decoder_clones = clone_many_times(decoder, opt.max_sent_l_targ)
  encoder_clones = clone_many_times(encoder, opt.max_sent_l_src)
  for i = 1, opt.max_sent_l_src do
    if encoder_clones[i].apply then
      encoder_clones[i]:apply(function(m) m:setReuse() end)
      if opt.prealloc == 1 then encoder_clones[i]:apply(function(m) m:setPrealloc() end) end
    end
  end
  for i = 1, opt.max_sent_l_targ do
    if decoder_clones[i].apply then
      decoder_clones[i]:apply(function(m) m:setReuse() end)
      if opt.prealloc == 1 then decoder_clones[i]:apply(function(m) m:setPrealloc() end) end
    end
  end

  local h_init = torch.zeros(opt.max_batch_l, opt.rnn_size)
  local attn_init = torch.zeros(opt.max_batch_l, opt.max_sent_l)
  if opt.gpuid >= 0 then
    h_init = h_init:cuda()
    attn_init = attn_init:cuda()
    cutorch.setDevice(opt.gpuid)
  end

  init_fwd_enc = {}
  init_bwd_enc = {}
  init_fwd_dec = {}
  init_bwd_dec = {}

  for L = 1, opt.num_layers do
    table.insert(init_fwd_enc, h_init:clone())
    table.insert(init_fwd_enc, h_init:clone())
    table.insert(init_bwd_enc, h_init:clone())
    table.insert(init_bwd_enc, h_init:clone())
  end
  if opt.gpuid2 >= 0 then
    cutorch.setDevice(opt.gpuid2)
  end
  if opt.input_feed == 1 then
    table.insert(init_fwd_dec, h_init:clone())
  end
  table.insert(init_bwd_dec, h_init:clone())
  for L = 1, opt.num_layers do
    table.insert(init_fwd_dec, h_init:clone())
    table.insert(init_fwd_dec, h_init:clone())
    table.insert(init_bwd_dec, h_init:clone())
    table.insert(init_bwd_dec, h_init:clone())
  end

  dec_offset = 3
  if opt.input_feed == 1 then
    dec_offset = dec_offset + 1
  end

  beam = require 's2sa.beam_modified'
  beam.init(opt)
  utf8 = require 'lua-utf8'
  total_queries = 0
  local encoder_grad_proto_adv = torch.zeros(1, opt.max_sent_l , opt.rnn_size)
  local encoder_bwd_grad_proto_adv = torch.zeros(1, opt.max_sent_l , opt.rnn_size)
  local context_proto_adv = torch.zeros(1, opt.max_sent_l, opt.rnn_size)
  if opt.gpuid >=0 then
    encoder_grad_proto_adv = encoder_grad_proto_adv:cuda()
    encoder_bwd_grad_proto_adv = encoder_bwd_grad_proto_adv:cuda()
    context_proto_adv = context_proto_adv:cuda()
  end

  function reset_state(state, batch_l, t)
    if t == nil then
      local u = {}
      for i = 1, #state do
        state[i]:zero()
        table.insert(u, state[i][{{1, batch_l}}])
      end
      return u
    else
      local u = {[t] = {}}
      for i = 1, #state do
        state[i]:zero()
        table.insert(u[t], state[i][{{1, batch_l}}])
      end
      return u
    end
  end
  function translate(source, source_features, target)
    local ind = target:size(1)
    local  output_table, bleu_score, attn, gold_score, all_sents, all_scores, all_attn = beam.gen_beam(model,
          opt.beam, opt.max_sent_l, source, source_features, target)
    function table_to_tensor(tab)
      ten = torch.Tensor(#tab,1)
      for i=1,#tab do
        ten[i] = tab[i]
      end
      return ten
    end   
    return table_to_tensor(output_table), bleu_score
  end


  function forward_backward(all_in_one, all_in_one_ind, source_l, source, source_features, target_l, target, target_out, type_ )
    total_queries = total_queries + 1
    local encoder_grads = encoder_grad_proto_adv[{{1, 1}, {1, source_l}}]
    local encoder_bwd_grads
    if opt.gpuid >= 0 then
      cutorch.setDevice(opt.gpuid)
    end
    local rnn_state_enc = reset_state(init_fwd_enc, 1, 0)
    local context = context_proto_adv[{{1, 1}, {1, source_l}}]
    -- forward prop encoder
    for t = 1, source_l do
      encoder_clones[t]:evaluate()
      local encoder_input = {source[t]}
      if data.num_source_features > 0 then
        append_table(encoder_input, source_features[t])
      end
      append_table(encoder_input, rnn_state_enc[t-1])
      local out = encoder_clones[t]:forward(encoder_input)
      rnn_state_enc[t] = out
      context[{{},t}]:copy(out[#out])
    end

    local rnn_state_enc_bwd
    if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
      cutorch.setDevice(opt.gpuid2)
      local context2 = context_proto2[{{1, 1}, {1, source_l}}]
      context2:copy(context)
      context = context2
    end
    -- copy encoder last hidden state to decoder initial state
    local rnn_state_dec = reset_state(init_fwd_dec, 1, 0)
    if opt.init_dec == 1 then
      for L = 1, opt.num_layers do
        rnn_state_dec[0][L*2-1+opt.input_feed]:copy(rnn_state_enc[source_l][L*2-1])
        rnn_state_dec[0][L*2+opt.input_feed]:copy(rnn_state_enc[source_l][L*2])
      end
    end
      
    -- forward prop decoder
    local preds = {}
    local attn_outputs = {}
    local decoder_input
    for t = 1, target_l do
      decoder_clones[t]:evaluate()
      local decoder_input
      if opt.attn == 1 then
        decoder_input = {target[t], context, table.unpack(rnn_state_dec[t-1])}
      else
        decoder_input = {target[t], context[{{}, source_l}], table.unpack(rnn_state_dec[t-1])}
      end
      
      local out = decoder_clones[t]:forward(decoder_input)
      local out_pred_idx = #out
      if opt.guided_alignment == 1 then
        out_pred_idx = #out-1
        table.insert(attn_outputs, out[#out])
      end
      local next_state = {}
      table.insert(preds, out[out_pred_idx])
      if opt.input_feed == 1 then
        table.insert(next_state, out[out_pred_idx])
      end
      for j = 1, out_pred_idx-1 do
        table.insert(next_state, out[j])
      end
      rnn_state_dec[t] = next_state
    end

      -- backward prop decoder
    encoder_grads:zero()

    local drnn_state_dec = reset_state(init_bwd_dec, 1)
    if opt.guided_alignment == 1 then
      attn_init:zero()
      table.insert(drnn_state_dec, attn_init[{{1, 1}, {1, source_l}}])
    end
    local loss = 0
    local loss_cll = 0
    for t =  target_location - 1,  target_location - 1 do--target_l, 1, -1 do
      --only backward for the target word
      local pred = generator:forward(preds[t])
      local input = pred
      local output = target_out[t]
      if type_ == 'none'  then
        local _ , sorted_ind = torch.sort(pred,2, true)
        local cnt_likely = 0
        local wrd = idx2word_targ[sorted_ind[1][opt.likely+cnt_likely]]
        local org_wrd = idx2word_targ[target_out[t][1]] 
        -- the two words are not substrings of each other, the new word is not in the translation already, does not have esxape characters, and is not stop word.
        while wrd:len() < 2 or  string.find(string.lower(wrd), string.lower(org_wrd)) ~= nil or  string.find(string.lower(org_wrd), string.lower(wrd)) ~= nil   or  target_out:eq(sorted_ind[1][opt.likely+cnt_likely]):sum() ~= 0 or stop_words_list[string.lower(wrd)] ~= nil  or    string.find(wrd,'&') ~= nil   do 
           cnt_likely = cnt_likely + 1
           if cnt_likely == 100 then
             break
           end
           wrd = idx2word_targ[sorted_ind[1][opt.likely+cnt_likely]]     
        end 
        likely_targets[t] = sorted_ind:t()[opt.likely + cnt_likely]:double()
        return 
      end
      if type_ ~= 'none' and not opt.controlled then             
        output = likely_targets[t]   
      end
      if opt.guided_alignment == 1 then
        input={input, attn_outputs[t]}
        output={output, norm_alignment[{{},{},t}]}
      end
      loss = loss + criterion:forward(input, output)
      local drnn_state_attn
      local dl_dpred
      if opt.guided_alignment == 1 then
        local dl_dpred_attn = criterion:backward(input, output)
        dl_dpred = dl_dpred_attn[1]
        drnn_state_attn = dl_dpred_attn[2] 
        loss_cll = loss_cll + cll_criterion:forward(input[1], output[1])
      else
        dl_dpred = criterion:backward(input, output)
      end        
      local dl_dtarget = generator:backward(preds[t], dl_dpred)
      local rnn_state_dec_pred_idx = #drnn_state_dec
      if opt.guided_alignment == 1 then
        rnn_state_dec_pred_idx = #drnn_state_dec-1
        drnn_state_dec[#drnn_state_dec]:add(drnn_state_attn)
      end
      drnn_state_dec[rnn_state_dec_pred_idx]:add(dl_dtarget)
      local decoder_input
      if opt.attn == 1 then
        decoder_input = {target[t], context, table.unpack(rnn_state_dec[t-1])}
      else
        decoder_input = {target[t], context[{{}, source_l}], table.unpack(rnn_state_dec[t-1])}
      end
      local dlst = decoder_clones[t]:backward(decoder_input, drnn_state_dec)
      -- accumulate encoder/decoder grads
      if opt.attn == 1 then
        encoder_grads:add(dlst[2])
      else
        encoder_grads[{{}, source_l}]:add(dlst[2])
      end
      drnn_state_dec[rnn_state_dec_pred_idx]:zero()
      if opt.guided_alignment == 1 then
        drnn_state_dec[#drnn_state_dec]:zero()
      end
      if opt.input_feed == 1 then
        drnn_state_dec[rnn_state_dec_pred_idx]:add(dlst[3])
      end
      for j = dec_offset, #dlst do
        drnn_state_dec[j-dec_offset+1]:copy(dlst[j])
      end
      --end
    end
    word_vec_layers[2].gradWeight[1]:zero()
    if opt.fix_word_vecs_dec == 1 then
      word_vec_layers[2].gradWeight:zero()
    end
    -- backward prop encoder
    if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
      cutorch.setDevice(opt.gpuid)
      local encoder_grads2 = encoder_grad_proto2[{{1, 1}, {1, source_l}}]
      encoder_grads2:zero()
      encoder_grads2:copy(encoder_grads)
      encoder_grads = encoder_grads2 -- batch_l x source_l x rnn_size
    end
    local drnn_state_enc = reset_state(init_bwd_enc, 1)
    if opt.init_dec == 1 then
      for L = 1, opt.num_layers do
        drnn_state_enc[L*2-1]:copy(drnn_state_dec[L*2-1])
        drnn_state_enc[L*2]:copy(drnn_state_dec[L*2])
      end
    end

    for t = source_l, 1, -1 do
      local encoder_input = {source[t]}
      if data.num_source_features > 0 then
        append_table(encoder_input, source_features[t])
      end
      append_table(encoder_input, rnn_state_enc[t-1])
      if opt.attn == 1 then
        drnn_state_enc[#drnn_state_enc]:add(encoder_grads[{{},t}])
      else
        if t == source_l then
          drnn_state_enc[#drnn_state_enc]:add(encoder_grads[{{},t}])
        end
      end
      local dlst = encoder_clones[t]:backward(encoder_input, drnn_state_enc)
      for j = 1, #drnn_state_enc do
        drnn_state_enc[j]:copy(dlst[j+1+data.num_source_features])
      end
        
      if type_ == 'adv' then 
        local flip_estimates = encoder_clones[t].modules[3].gradInput:clone()
        local insert_estimates = encoder_clones[t].modules[3].gradInput:clone()
        local del_estimates =  encoder_clones[t].modules[3].gradInput:clone()
        local swap_estimates = encoder_clones[t].modules[3].gradInput:clone()
        update_flip(torch.squeeze(source[t]), flip_estimates, ctrl1)
        update_insert(torch.squeeze(source[t]), insert_estimates, ctrl1)
        update_delete(torch.squeeze(source[t]), del_estimates, ctrl1)
        update_swap(torch.squeeze(source[t]), swap_estimates, ctrl1)
        remove_special_chars(source[t], flip_estimates, ctrl2)
        remove_special_chars(source[t], insert_estimates, ctrl2)
        remove_special_chars(source[t], del_estimates, ctrl2)
        remove_special_chars(source[t], swap_estimates, ctrl2)
        flip_estimates[{{},{1,6}}]:fill(ctrl2)
        insert_estimates[{{},{1,6}}]:fill(ctrl2)
        del_estimates[{{},{1,6}}]:fill(ctrl2)
        swap_estimates[{{},{1,6}}]:fill(ctrl2)
        for x=1, flip_estimates:size(1) do
          if (source[t][1][x] >= 1 and  source[t][1][x] <= 6)  then               
            flip_estimates[x]:fill(ctrl2)
            insert_estimates[x]:fill(ctrl2)
            del_estimates[x]:fill(ctrl2)
            swap_estimates[x]:fill(ctrl2)
          end
        end
        if not opt.controlled then
          local a_,b_ = torch.min(flip_estimates,2) 
          all_in_one[1][t]:copy(a_)
          all_in_one_ind[1][t]:copy(b_)
          a_,b_ = torch.min(insert_estimates,2)
          all_in_one[2][t]:copy(a_)
          all_in_one_ind[2][t]:copy(b_)
          a_,b_ = torch.min(del_estimates,2)
          all_in_one[3][t]:copy(a_)
          a_,b_ = torch.min(swap_estimates,2)
          all_in_one[4][t]:copy(a_) 
        else
          local a_,b_ = torch.max(flip_estimates,2)
          all_in_one[1][t]:copy(a_)
          all_in_one_ind[1][t]:copy(b_)
          a_,b_ = torch.max(insert_estimates,2)
          all_in_one[2][t]:copy(a_)
          all_in_one_ind[2][t]:copy(b_)
          a_,b_ = torch.max(del_estimates,2)
          all_in_one[3][t]:copy(a_)
          a_,b_ = torch.max(swap_estimates,2)
          all_in_one[4][t]:copy(a_)
        end
      end
    end

    word_vec_layers[1].gradWeight[1]:zero()
    if opt.fix_word_vecs_enc == 1 then
      word_vec_layers[1].gradWeight:zero()
    end

    return loss   
  end

  function print_linked_list(list)
    local tmp = list
    print('***')
    while tmp do
      --print(tmp.value[1], tmp.value[2], tmp.value[3], tmp.value[4], tmp.value[5])
      change_types_total[tmp.value[6]] = change_types_total[tmp.value[6]] + 1
      tmp = tmp.next
    end
  end

  function beam_search_mix( all_in_one_total, all_in_one_ind_total,  source,  instance, source_l,  gold,  bleu_s_, target, source_features , target_l, target_out , bleu_unk)
    local len = 1
    local max_change = math.max(math.floor(count_chars(torch.squeeze(source[{{},{instance},{}}],2)) * opt.perc_) ,1) 
    local beam_size = opt.beam_size
    changes_type = {}
    changes_type['flip'] = 1
    changes_type['insert'] = 2
    changes_type['del'] = 3
    changes_type['swap'] = 4
    local states_ = initialize_beam(all_in_one_total, all_in_one_ind_total, beam_size, {}, source, instance, opt.controlled) 
    if #states_ == 0 then return false end
    table.sort(states_, compare)
    states = {}
    for i=1, beam_size do
      table.insert(states, states_[i])
    end         
    while len <= max_change  do           
      local cn = 1
      local added_paths = {}
      local adv_exmp = source[{{},{instance},{}}]:clone()
      while cn <= beam_size and cn <= #states  do
        local n__ = states[cn]
        local n_ = reverse_list(n__)
        local tmp = n_
        local ccnt = 0
        while tmp do
          if tmp.value[6] == 1 then
            adv_exmp[tmp.value[1]][1][tmp.value[2]] = tmp.value[3]
          elseif tmp.value[6] == 2 then
            shift_right(adv_exmp[tmp.value[1]][1], tmp.value[3], tmp.value[2])
          elseif tmp.value[6] == 3 then
            shift_left(adv_exmp[tmp.value[1]][1], tmp.value[2])
          else 
            swap(adv_exmp[tmp.value[1]][1], tmp.value[2])
          end
          tmp =  tmp.next
          ccnt = ccnt + 1
        end
        T_adv, a_bleu = translate(adv_exmp, source_features, torch.squeeze(gold) )   
        --print(original_targ:size())         
        if (not opt.controlled and torch.eq(T_adv, original_word_id):sum() == 0 and torch.eq(T_adv, likely_targets[target_location - 1][1]):sum() > 0) or 
                (opt.controlled  and torch.eq(T_adv, original_word_id):sum() == 0) then  
          local _,own_bleu = translate(adv_exmp, source_features,torch.squeeze(target))   
          print(idx2word_targ[target[target_location][1]], idx2word_targ[likely_targets[target_location - 1][1]]) 
          --print('original ' .. print_ex(torch.squeeze(source[{{},{instance},{}}]), idx2char)) 
          print('adversarial :::: ' .. print_ex(torch.squeeze(adv_exmp), idx2char))
          print('new translation :::: ' .. print_seq(T_adv, idx2word_targ))
          --print('original translation ' .. print_seq(target, idx2word_targ))
          return true , ccnt              
        else
          --don't extend the beam if it's the last character in budget
          if len < max_change then
            zero_table(grad_params, 'zero')
            loss = forward_backward(all_in_one_total, all_in_one_ind_total, source_l , adv_exmp, source_features, target_l, target , target_out , 'adv')
            added_paths = extend_beam(all_in_one_total, all_in_one_ind_total, beam_size,  added_paths, n__, adv_exmp, source[{{},{instance},{}}], loss, opt.controlled)
          end
          adv_exmp = source[{{},{instance},{}}]:clone()
        end
        cn = cn + 1
      end           
      states = {}
      table.sort(added_paths, compare)           
      table.insert(states, added_paths[1])
      local j = 2
      while #states < beam_size and j < #added_paths do
         --check that duplicated paths, which will create equal advesarial examples, are not added
        if not if_exists(states, added_paths[j], source[{{},{instance},{}}] ) then
          table.insert(states, added_paths[j])
        end   
        j = j + 1
      end    
      len = len + 1
    end
    return false , 0  
  end
     
  idx2word_targ = idx2key(opt.targ_dict)
  idx2word_src = idx2key(opt.src_dict)
  word_src = src_words(opt.src_dict) 
  
  char2idx = flip_table(idx2key(opt.char_dict))  
  word_to_id = flip_table(idx2word_targ)
  target_location  = 0 
  change_types_total = torch.Tensor(4):zero()
  
  local total = 0 
  local all_docs = 0
  local bleu_s = 0
  g_bleu = 0
  g_bleu_2 = 0 
  adv_bleu = 0 
  local ret_tens
  g_cnt = 0
  for i = 1, data:size() do
    local d = data[i]
    local target, target_out, nonzeros, source = d[1], d[2], d[3], d[4]
    local batch_l, target_l, source_l = d[5], d[6], d[7]
    local source_features = d[9]
    local alignment = d[10]
    local norm_alignment
    if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
      cutorch.setDevice(opt.gpuid)
    end
    for instance=1, batch_l do  
      local gold = target[{{},{instance}}]    
      local ret_tens, bleu_s_ =  translate(source[{{},{instance},{}}], source_features, torch.squeeze(gold))

      local unk_targ = ret_tens:clone()
      local cnt_wrd = 0
      for wrd=2, ret_tens:size(1) do
         if ret_tens[wrd][1] ~= 4 then
            cnt_wrd = cnt_wrd + 1
         end
      end
      local search_all = {}
      local found_good_word = false
      rp = torch.randperm(cnt_wrd)
      for r=1,rp:size(1) do
        target_location = 1 + rp[r] 
        local k = idx2word_targ[ret_tens[target_location][1]]
         -- a candidate word is not a stop word, is not escaped in preprocessing and the length is bigger than 1
        if stop_words_list[string.lower(k)] == nil and string.find(k,'&') == nil and k:len() > 1 then
          found_good_word = true
          original_word_id = ret_tens[target_location][1]
          break
        end
      end
      
      local new_target_l = cnt_wrd + 2
      local new_target_out = torch.cat(ret_tens:sub(2,-1),torch.Tensor(1,1):fill(1),1):cuda()
      local new_target = ret_tens:cuda()
      
      if new_target_l <=  opt.max_sent_l and   cnt_wrd > 0 and  source_l > 1  and found_good_word then 
        all_docs = all_docs + 1
        target_l = new_target_l
        
        local all_in_one_total = torch.Tensor(4,source_l, source:size(3)):zero()
        local all_in_one_ind_total = torch.Tensor(4,source_l, source:size(3)):zero()
        likely_targets = new_target_out:clone()  
        likely_targets:zero()
        if not opt.controlled then
          zero_table(grad_params, 'zero')
          forward_backward(nil, nil, source_l , source[{{},{instance},{}}], source_features, target_l, new_target , new_target_out,  'none')
          unk_targ[target_location][1] = likely_targets[target_location - 1][1] 
        else
          unk_targ[target_location][1] = 2            
        end
        local bleu_unk = get_bleu(torch.squeeze(unk_targ), torch.squeeze(new_target):sub(2,-2))
        zero_table(grad_params, 'zero')
        forward_backward(all_in_one_total, all_in_one_ind_total, source_l , source[{{},{instance},{}}], source_features, target_l, new_target , new_target_out,  'adv')
        local found, cccc = beam_search_mix( all_in_one_total, all_in_one_ind_total,  source,  instance, source_l, gold,  bleu_s_, new_target, source_features, target_l, new_target_out,   bleu_unk)
      end
    end
  end
end
function get_layer(layer)
  if layer.name ~= nil then
    if layer.name == 'word_vecs_dec' then
      table.insert(word_vec_layers, layer)
    elseif layer.name == 'word_vecs_enc' then
      table.insert(word_vec_layers, layer)
    elseif layer.name == 'charcnn_enc' or layer.name == 'mlp_enc' then
      local p, gp = layer:parameters()
      for i = 1, #p do
        table.insert(charcnn_layers, p[i])
        table.insert(charcnn_grad_layers, gp[i])
      end
    end
  end
end

function main()
  -- parse input params
  opt = cmd:parse(arg)
  if opt.controlled == 0 then
     opt.controlled = false
  else
    opt.controlled = true
  end
  if opt.controlled then
     ctrl1 = -1e5
     ctrl2 = -1e9
  else
     ctrl1 = 1e5
     ctrl2 = 1e9
  end
  idx2char = idx2key(opt.char_dict)
  torch.manualSeed(opt.seed)
  stop_words_list =  stop_words('WSLT/stop.words') 
 

  if opt.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    if opt.gpuid2 >= 0 then
      print('using CUDA on second GPU ' .. opt.gpuid2 .. '...')
    end
    require 'cutorch'
    require 'cunn'
    if opt.cudnn == 1 then
      print('loading cudnn...')
      require 'cudnn'
    end
    cutorch.setDevice(opt.gpuid)
    cutorch.manualSeed(opt.seed)
  end

  -- Create the data loader class.
  print('loading data...')
  test_data = data.new(opt, opt.test_data_file)
  print('done!')
  print(string.format('Source vocab size: %d, Target vocab size: %d',
      test_data.source_size, test_data.target_size))
  opt.max_sent_l_src = test_data.source:size(2)
  opt.max_sent_l_targ = test_data.target:size(2)
  opt.max_sent_l = math.max(opt.max_sent_l_src, opt.max_sent_l_targ)
  if opt.max_batch_l == '' then
    opt.max_batch_l =  test_data.batch_l:max()--train_data.batch_l:max()
  end

  if opt.use_chars_enc == 1 or opt.use_chars_dec == 1 then
    opt.max_word_l = test_data.char_length
  end
  print(string.format('Source max sent len: %d, Target max sent len: %d',
      test_data.source:size(2), test_data.target:size(2)))


  -- Enable memory preallocation - see memory.lua
  preallocateMemory(opt.prealloc)
  assert(path.exists(opt.saved_model), 'checkpoint path invalid')
  print('loading ' .. opt.saved_model .. '...')
  local checkpoint = torch.load(opt.saved_model)
  local model, model_opt = checkpoint[1], checkpoint[2]  
  encoder_ = model[1]
  -- Replace the lookup table with a linear layer which can collect gradients too. Add a one-hot layer too. Won't be able to do batching
  encoder = make_lstm(test_data, opt, 'enc', opt.use_chars_enc)
  local p, gp = encoder:getParameters()
  local p_, gp_ = encoder_:getParameters()    
  p:copy(p_)
  encoder.modules[3].weight:copy(encoder_.modules[2].weight:t())
  decoder = model[2]
  generator = model[3]
  _, criterion = make_generator(test_data, opt)
  opt.model = {{encoder, decoder, generator},opt}
  if opt.guided_alignment == 1 then
    cll_criterion = criterion
    criterion = nn.ParallelCriterion()
    criterion:add(cll_criterion, (1-opt.guided_alignment_weight))
    -- sum of alignment weight reconstruction loss over all input/output pair; averaged
    criterion:add(nn.MSECriterion(), opt.guided_alignment_weight)
  end
  layers = {encoder, decoder, generator}
  if opt.optim ~= 'sgd' then
    layer_etas = {}
    optStates = {}

    if opt.layer_lrs:len() > 0 then
      local stringx = require('pl.stringx')
      local lr_strings = stringx.split(opt.layer_lrs, ',')
      if #lr_strings ~= #layers then error('1 learning rate per layer expected') end
      for i = 1, #lr_strings do
        local lr = tonumber(stringx.strip(lr_strings[i]))
        if not lr then
          error(string.format('malformed learning rate: %s', lr_strings[i]))
        else
          layer_etas[i] = lr
        end
      end
    end

    for i = 1, #layers do
      layer_etas[i] = layer_etas[i] or opt.learning_rate
      optStates[i] = {}
    end
  end

  if opt.gpuid >= 0 then
    for i = 1, #layers do
      if opt.gpuid2 >= 0 then
        if i == 1 or i == 4 then
          cutorch.setDevice(opt.gpuid) --encoder on gpu1
        else
          cutorch.setDevice(opt.gpuid2) --decoder/generator on gpu2
        end
      end
      layers[i]:cuda()
    end
    if opt.gpuid2 >= 0 then
      cutorch.setDevice(opt.gpuid2) --criterion on gpu2
    end
    criterion:cuda()
  end

  -- these layers will be manipulated during training
  word_vec_layers = {}
  if opt.use_chars_enc == 1 then
    charcnn_layers = {}
    charcnn_grad_layers = {}
  end
  encoder:apply(get_layer)
  decoder:apply(get_layer)
  generate_adv(test_data)
  --setup()
end

main()
