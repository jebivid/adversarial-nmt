function update_flip(instance, grad, ctrl)
    local grad_ = torch.CudaTensor(grad:size(1),1):fill(0)
    for i=1, instance:size(1) do
       if instance[i] == 4 then break end
        grad_[i] = grad[i][instance[i]]
    end
    grad:csub(grad_:expand(grad:size()))
    -- set the no change estimate (a -> a) to a high/low value depending on the attack
    --for i=1, instance:size(1) do
    --    if instance[i] == 4 then break end
    --    grad[i][instance[i]] = ctrl
    --end
end

function update_swap(instance, grad, ctrl)
    local grad_ = torch.CudaTensor(grad:size(1),1):fill(0)
    for i=1, instance:size(1) do
       if instance[i] == 4 then break end
       grad_[i] = grad[i][instance[i]]
    end
    grad:csub(grad_:expand(grad:size()))
    local grad2_ = torch.CudaTensor(grad:size(1),1):fill(ctrl)
    for i=2, instance:size(1)-1  do
      if instance[i+1] == 4 then  break end
      grad2_[i] = grad[i][instance[i+1]] + grad[i+1][instance[i]]  
    end
    grad:fill(ctrl)
    -- no flip in swap or delete. So we need to only set one column to hold the estimate for possible swaps or deletes
    grad[{{},{7}}]:copy(grad2_)
end


function update_insert(instance, grad, ctrl)
    local grad_ = torch.CudaTensor(grad:size(1),1):fill(0)
    local cnt = 0 
    for i=1, instance:size(1) do
       if instance[i] == 4 then break end
       grad_[i] = grad[i][instance[i]]
    end
    grad:csub(grad_:expand(grad:size()))
    local grad2_ = torch.CudaTensor(grad:size(1),1):fill(0)
    for i=2, instance:size(1)  do
      if instance[i] == 4 then break end
      for j=i+1, math.min(i+1, instance:size(1)) do 
        if instance[j] == 4 then break end
        grad2_[i] = grad2_[i] + grad[j][instance[j-1]]
      end
    end
    grad:add(grad2_:expand(grad:size()))  
    --for i=1, instance:size(1) do
    --   if instance[i] == 4 then break end
    --    grad[i][instance[i]] = ctrl
    --end
end

function update_delete(instance, grad, ctrl)
    local grad_clone = grad:clone()
    local grad_ = torch.CudaTensor(grad:size(1),1):fill(0)
    for i=2, instance:size(1) do
       if instance[i] == 4 then break end
       grad_[i] = grad_clone[i][instance[i]]
    end
    grad_clone:csub(grad_:expand(grad:size()))
    local grad2_ = torch.CudaTensor(grad:size(1),1):fill(0)
    for i=2, instance:size(1)-1  do
      if instance[i+1] == 4 then grad2_[i] = ctrl  break end
      for j=i+1, math.min(i+2, instance:size(1)) do 
        if instance[j] == 4 then break end
        grad2_[i] = grad2_[i] + grad_clone[j-1][instance[j]]
      end
    end 
    grad:fill(ctrl)
    grad[{{},{7}}]:copy(grad2_)
end

function shift_right(instance,  what, from)
    if from ~= 35 then
      instance:sub(from+1,-1):copy(instance:sub(from,-2))
      instance[from] = what
    end
end

function shift_left(instance, from)
    if instance[3] ~= 4 then
      if from ~= 35 then
         instance:sub(from,-2):copy(instance:sub(from+1,-1))
         instance[-1]=1
      end
    end
end

function swap(instance, from)
    if instance[3] ~= 4 then
      if from ~= 35 then
         instance[from+1],  instance[from] = instance[from], instance[from+1] 
      end
    end
end

