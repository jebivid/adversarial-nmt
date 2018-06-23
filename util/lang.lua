function idx2key(file)
   local f = io.open(file,'r')
   local t = {}
   for line in f:lines() do
      local c = {}
      for w in line:gmatch'([^%s]+)' do
        table.insert(c, w)
      end
      t[tonumber(c[2])] = c[1]
   end
   return t
end

function src_words(file)
    local f = io.open(file,'r')
    local t = {}
    for line in f:lines() do
      local c = {}
      for w in line:gmatch'([^%s]+)' do
        table.insert(c, w)
      end
      t[c[1]] = true
    end
    return t
end

function flip_table(u)
   local t = {}
   for key, value in pairs(u) do
      t[value] = key
   end
   return t
end

