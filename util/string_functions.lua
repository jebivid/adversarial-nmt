function remove_special_chars(instance, M, ctrl)
    -- used python2 to preprocess which escapes special characters.
    -- preventing manipulations of some of these entities to be considered as legit adversarial manipulations
    -- (for targeted and controlled attacks)
    local st_ = string.find(print_word(torch.squeeze(instance), idx2char),'&apos')
    if st_ ~= nil then
      M:sub(st_+1,st_+5):fill(ctrl)
    end
    local st_ = string.find(print_word(torch.squeeze(instance), idx2char),'&apos;')
    if st_ ~= nil then
      M:sub(st_+1,st_+6):fill(ctrl)
    end
    local st_ = string.find(print_word(torch.squeeze(instance), idx2char),'&quot')
    if st_ ~= nil then
       M:sub(st_+1,st_+5):fill(ctrl)
    end
    local st_ = string.find(print_word(torch.squeeze(instance), idx2char),'&quot;')
    if st_ ~= nil then
      M:sub(st_+1,st_+6):fill(ctrl)
    end
end

function print_seq(seq, idx)
   local str=''
   for i=1,seq:size(1) do
      if seq[i][1] > 6 then
         if str:len() > 0 then
            str = str .. ' ' .. idx[seq[i][1]]
         else
            str = idx[seq[i][1]]
         end
      end
   end
   return str
end

function print_ex(instance, idx2char)
   local str =''
   if instance:dim() > 1 then
       local source_l = instance:size(1)
       for t=1, source_l do
          for i=2, instance:size(2) do
             if instance[t][i] > 6 then
               str = str .. idx2char[instance[t][i]]
             end
          end
          str = str .. ' '
       end
   else
      for i=1, instance:size(1) do
         if instance[i] > 6 then
            str = str .. idx2char[instance[i]]
         end
      end
      str = str .. ' '
   end
   return str
end

function count_chars(instance)
   local cnt = 0
   if instance:dim() > 1 then
      local source_l = instance:size(1)
      for t=1, source_l do
        for i=1, instance:size(2) do
          if instance[t][i] > 6  then
            cnt = cnt + 1
          end
          if instance[t][i] == 4 then break end
        end
      end
   else
      for i=1, instance:size(1) do
          if instance[i] > 6  then
            cnt = cnt + 1
          end
          if instance[i] == 4 then break end
      end
   end
   return cnt
end
function print_word(instance, idx2char)
   local str= ''
   for i=2,instance:size(1) do
       if instance[i] == 4 then break end
       str = str .. idx2char[instance[i]]
   end
   return str
end




