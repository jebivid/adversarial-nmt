function compare(a,b)
  return a.val >  b.val
end

function reverse_list(list)
   local new_list = nil
   local tmp = list
   while tmp do
      new_list = {next = new_list, value = tmp.value, dist=tmp.dist, val = tmp.val}
      tmp = tmp.next
   end
   return new_list
end

--used to avoid duplicate paths in our search
function if_exists(states, candidate, instance)
   local instance_ = instance:clone()
   local instance_2 = instance:clone()
   local n_ = reverse_list(candidate)
   local tmp = n_
   while tmp do
      if   tmp.value[6] == 1 then
         instance_2[tmp.value[1]][1][tmp.value[2]] = tmp.value[3]
      elseif tmp.value[6] == 2 then
         shift_right(instance_2[tmp.value[1]][1], tmp.value[3], tmp.value[2])
      elseif tmp.value[6] == 3 then
         shift_left(instance_2[tmp.value[1]][1],  tmp.value[2])
      else
         swap(instance_2[tmp.value[1]][1],  tmp.value[2])
      end
      tmp =  tmp.next
   end
   for i=1,#states do
      local n__ = states[i]
      n_ = reverse_list(n__)
      tmp = n_
      while tmp do
         if   tmp.value[6] == 1 then
            instance_[tmp.value[1]][1][tmp.value[2]] = tmp.value[3]
         elseif tmp.value[6] == 2 then
            shift_right(instance_[tmp.value[1]][1], tmp.value[3], tmp.value[2])
         elseif tmp.value[6] == 3 then
            shift_left(instance_[tmp.value[1]][1],  tmp.value[2])
         else
            swap(instance_[tmp.value[1]][1],  tmp.value[2])
         end
         tmp =  tmp.next
      end
      if torch.all(torch.eq(instance_, instance_2)) then return true end
   end
   return false
end

--used to avoid creating previously seen aversarial examples
function full_search_linked_list(instance, source, list, a , b , ind, type_)
   adv_exmp = source:clone()
   local tmp = list
   local num = 0
   local instance_ = instance:clone()
   if type_ == 'flip' then
     instance_[a][1][b] = ind
   elseif type_ == 'insert' then
     shift_right(instance_[a][1], ind, b)
   elseif type_ == 'del' then
     shift_left(instance_[a][1],b)
   else
     swap(instance_[a][1], b)
   end

   while tmp.next do
      if tmp.value[6] == 1 then
         adv_exmp[tmp.value[1]][1][tmp.value[2]] = tmp.value[3]
      elseif tmp.value[6] == 2 then
         shift_right(adv_exmp[tmp.value[1]][1], tmp.value[3], tmp.value[2])
      elseif tmp.value[6] == 3 then
         shift_left(adv_exmp[tmp.value[1]][1], tmp.value[2])
      else
         swap(adv_exmp[tmp.value[1]][1], tmp.value[2])
      end
      if torch.all(torch.eq(torch.squeeze(adv_exmp) , torch.squeeze(instance_))) then return true end
      tmp = tmp.next
   end
   return false
end

--function to disallow more than 2 or k times of manipulation per word
function exhuasted_word(list , a)
   local tmp = list
   local count_tbl = {}
   local cnt = 1
   while tmp do
      local word = tmp.value[1]
      if count_tbl[word] == nil then
         count_tbl[word] = 1
      else
         count_tbl[word] = count_tbl[word] + 1
      end
      tmp = tmp.next
   end
   if count_tbl[a] == 2 then
      return true
   end
   return false
end

function initialize_beam(all_in_one, all_in_one_ind, beam_size,  states, source, instance, untargeted)
   local aio  =  torch.reshape(all_in_one, all_in_one:size(1) * all_in_one:size(2) * all_in_one:size(3),1)
   
   local aval,aind = torch.sort(aio,1, untargeted)
   local b = 1
   --print(aval[b][1], untargeted) 
   while b <= beam_size and aval[b][1] ~= 0 do
      if untargeted and aval[b][1]  <= -1e5 then break end
      if not untargeted and aval[b][1]  >= 1e5 then break end
      local type_ = math.min(math.floor(aind[b][1] / ( all_in_one:size(2) *  all_in_one:size(3))) + 1, 4)
      if aind[b][1] % ( all_in_one:size(2) *  all_in_one:size(3)) == 0 then
         type_ = aind[b][1] / ( all_in_one:size(2) *  all_in_one:size(3))
      end
      local row = math.floor((aind[b][1] - ((type_-1) *( all_in_one:size(2) *  all_in_one:size(3)) )) / all_in_one:size(3)) + 1
      local col =  (aind[b][1] - ((type_-1) *( all_in_one:size(2) *  all_in_one:size(3)) )) % all_in_one:size(3)
      if col == 0 then
         col = all_in_one:size(3)
         row = (aind[b][1] - ((type_-1) *( all_in_one:size(2) *  all_in_one:size(3)) )) / all_in_one:size(3)
      end
      local adv_exmp = source[{{},{instance},{}}]:clone()
      if type_ == 1 then
         adv_exmp[row][1][col] = all_in_one_ind[type_][row][col]
      elseif type_ == 2 then
         shift_right(adv_exmp[row][1],  all_in_one_ind[type_][row][col], col)
      elseif type_ == 3 then
         shift_left(adv_exmp[row][1], col)
      else
         swap(adv_exmp[row][1], col)
      end
      -- check if the new word is not already in the vocabualry
      if not word_src[print_word(torch.squeeze(adv_exmp[row][1]), idx2char)] then
         local next_ = torch.Tensor(6):zero()
         next_[1] = row
         next_[2] = col
         if type_ ~= 3 and type_ ~= 4 then
            next_[3] = all_in_one_ind[type_][row][col]
         else
            next_[3] = 0
         end
         next_[4] = all_in_one[type_][row][col]
         next_[5] = torch.squeeze(source[{{},{instance},{}}],2)[row][col]
         next_[6] = type_
         local node = nil
         node = {next = node, value = next_, dist=0, val = next_[4], loss=0}
         table.insert(states, node)
         b = b + 1
      else
         b  = b + 1
         beam_size = beam_size  + 1
      end
   end
   return states
end
function extend_beam (all_in_one, all_in_one_ind, beam_size,  paths, n_, adv_exmp_in, source_, losses, untargeted)
   local aio  =  torch.reshape(all_in_one, all_in_one:size(1) * all_in_one:size(2) * all_in_one:size(3),1)
   local aval,aind = torch.sort(aio,1,untargeted)
   local b=1
   local temp_beam = beam_size
   while b <= temp_beam and aval[b][1] ~= 0  do
      if untargeted and aval[b][1]  <= -1e5 then break end
      if not untargeted and aval[b][1]  >= 1e5 then break end
      local type_ = math.min(math.floor(aind[b][1] / ( all_in_one:size(2) *  all_in_one:size(3))) + 1, 4)
      if aind[b][1] % ( all_in_one:size(2) *  all_in_one:size(3)) == 0 then
         type_ = aind[b][1] / ( all_in_one:size(2) *  all_in_one:size(3))
      end
      local row = math.floor((aind[b][1] - ((type_-1) *( all_in_one:size(2) *  all_in_one:size(3)) )) / all_in_one:size(3)) + 1
      local col =  (aind[b][1] - ((type_-1) *( all_in_one:size(2) *  all_in_one:size(3)) )) % all_in_one:size(3)
      if col == 0 then
         col = all_in_one:size(3)
         row = (aind[b][1] - ((type_-1) *( all_in_one:size(2) *  all_in_one:size(3)) )) / all_in_one:size(3)
      end
      local adv_exmp = adv_exmp_in:clone()
      if type_ == 1 then
         adv_exmp[row][1][col] =  all_in_one_ind[type_][row][col]
      elseif type_ == 2 then
         shift_right(adv_exmp[row][1],  all_in_one_ind[type_][row][col], col)
      elseif type_ == 3 then
         shift_left(adv_exmp[row][1], col)
      else
         swap(adv_exmp[row][1], col)
      end
      -- check if the new word is not in the vocabulary and the created examples is not eaual to the original
      if (not word_src[print_word(torch.squeeze(adv_exmp[row][1]), idx2char)]) and (not torch.all(torch.eq(torch.squeeze(adv_exmp), torch.squeeze(source_)))) then
         local next_ = torch.Tensor(6):zero()
         next_[1] = row
         next_[2] = col
         if type_ ~= 3 and type_ ~= 4 then
            next_[3] = all_in_one_ind[type_][row][col]
         else
            next_[3] = 0
         end
         next_[4] = all_in_one[type_][row][col]
         next_[5] = adv_exmp_in[row][1][col]
         next_[6] = type_
         -- check if this change would not create an adversarial example previously seen
         if  not full_search_linked_list(adv_exmp_in, source_ ,  n_, row, col, next_[3], type_)  then
            local val_loss = losses--[type_]
            local node = {next = n_, value = next_, dist = diff_, val = val_loss + next_[4], loss=val_loss}
            table.insert(paths, node )
         else
            if temp_beam < 2 * opt.beam_size then
               temp_beam = temp_beam + 1
            end
         end
         b = b + 1
      else
         b = b + 1
         temp_beam = temp_beam + 1
      end
   end
   return paths
end

