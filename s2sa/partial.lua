require 's2sa.OneHot'
require 's2sa.LinearNoBias'

function make_partial(data,opt)
  local inputs = {}
  local outputs = {}
  table.insert(inputs, nn.Identity()())
  local lookup = nn.LinearNoBias( data.char_size , opt.char_vec_size)
  --local char_vecs = nn.View(1, -1, opt.char_vec_size):setNumInputDims(2)(lookup(OneHot(data.char_size)(inputs[1])))
  local char_vecs = lookup(OneHot(data.char_size)(inputs[1]))
  table.insert(outputs, char_vecs)
  return nn.gModule(inputs, outputs) 
end
