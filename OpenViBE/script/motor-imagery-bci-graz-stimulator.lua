---@diagnostic disable: lowercase-global, undefined-global
function initialize(box)
    -- 加载刺激代码文件，这个文件包含了所有的刺激标识符
    dofile(box:get_config("${Path_Data}") .. "/plugins/stimulation/lua-stimulator-stim-codes.lua")

    -- 获取实验设置项
    -- 获取试验的总次数
    number_of_trials = box:get_setting(2)
    
    -- 获取分类标识符
    first_class = _G[box:get_setting(3)]
    second_class = _G[box:get_setting(4)]
    third_class = _G[box:get_setting(5)]
    fourth_class = _G[box:get_setting(6)]

    -- 获取基线阶段的持续时间（秒）
    baseline_duration = box:get_setting(7)
    -- 获取等待音频提示的时间（秒）
    wait_for_beep_duration = box:get_setting(8)
    -- 获取等待提示符号显示的时间（秒）
    wait_for_cue_duration = box:get_setting(9)
    -- 获取显示提示符号的持续时间（秒）
    display_cue_duration = box:get_setting(10)
    -- 获取反馈的持续时间（秒）
    feedback_duration = box:get_setting(11)
    -- 获取试验结束时的最短持续时间（秒）
    end_of_trial_min_duration = box:get_setting(12)
    -- 获取试验结束时的最长持续时间（秒）
    end_of_trial_max_duration = box:get_setting(13)

    -- 初始化随机数生成器
    math.randomseed(os.time())

    -- 确保试验次数是整数
    number_of_trials = math.floor(number_of_trials)
    
    -- 检查试验次数是否合理
    if number_of_trials < 4 then
        box:log("Warning", "试验次数少于4,可能不足以进行有效的四分类实验")
        number_of_trials = 4 -- 设置最小值为4
    end

    -- 创建一个均衡的序列，确保每个类别有大致相等的试验次数
    sequence = {}
    
    -- 计算每个类别应有的试验次数（向下取整）
    local trials_per_class = math.floor(number_of_trials / 4)
    
    -- 先添加相等次数的四种类别
    for i = 1, trials_per_class do
        table.insert(sequence, first_class)
        table.insert(sequence, second_class)
        table.insert(sequence, third_class)
        table.insert(sequence, fourth_class)
    end
    
    -- 处理剩余的试验（如果有）
    local remaining = number_of_trials - (trials_per_class * 4)
    if remaining > 0 then
        local classes = {first_class, second_class, third_class, fourth_class}
        for i = 1, remaining do
            -- 随机选择一个类别添加到序列中
            table.insert(sequence, classes[math.random(1, 4)])
        end
    end

    -- 使用Fisher-Yates洗牌算法随机化序列顺序
    for i = #sequence, 2, -1 do
        local j = math.random(i)
        sequence[i], sequence[j] = sequence[j], sequence[i]
    end
    
    -- 日志记录确认序列长度
    box:log("Info", string.format("实验序列已创建，共包含%d个试验", #sequence))
end

function process(box)
    -- 初始化时间变量
    local t = 0

    -- 发送实验开始的刺激
    box:send_stimulation(1, OVTK_StimulationId_ExperimentStart, t, 0)
    -- 时间增加 5 秒
    t = t + 5
    -- 发送基线开始的刺激
    box:send_stimulation(1, OVTK_StimulationId_BaselineStart, t, 0)
    -- 时间增加基线阶段的持续时间
    t = t + baseline_duration
    -- 发送基线结束的刺激
    box:send_stimulation(1, OVTK_StimulationId_BaselineStop, t, 0)
    -- 时间增加 5 秒
    t = t + 5

    -- 开始进行试验
    for i = 1, #sequence do
        -- 发送试验开始的刺激
        box:send_stimulation(1, OVTK_GDF_Start_Of_Trial, t, 0)
        -- 显示十字标记（用于标示开始位置）
        box:send_stimulation(1, OVTK_GDF_Cross_On_Screen, t, 0)
        -- 时间增加等待音频提示的时间
        t = t + wait_for_beep_duration

        -- 发送音频提示
        box:send_stimulation(1, OVTK_StimulationId_Beep, t, 0)
        -- 时间增加等待提示符号显示的时间
        t = t + wait_for_cue_duration

        -- 根据随机序列发送相应的提示符号
        box:send_stimulation(1, sequence[i], t, 0)
        -- 时间增加显示提示符号的持续时间
        t = t + display_cue_duration

        -- 发送反馈刺激
        box:send_stimulation(1, OVTK_GDF_Feedback_Continuous, t, 0)
        -- 时间增加反馈的持续时间
        t = t + feedback_duration

        -- 发送试验结束的刺激
        box:send_stimulation(1, OVTK_GDF_End_Of_Trial, t, 0)
        -- 时间增加一个随机的试验结束持续时间
        t = t + math.random(math.floor(end_of_trial_min_duration), math.floor(end_of_trial_max_duration))
    end

    -- 发送实验结束的刺激
    box:send_stimulation(1, OVTK_GDF_End_Of_Session, t, 0)
    -- 时间增加 5 秒
    t = t + 5
    -- 发送训练结束的刺激
    box:send_stimulation(1, OVTK_StimulationId_Train, t, 0)
    -- 时间增加 1 秒
    t = t + 1
    -- 用于停止采集过程
    box:send_stimulation(1, OVTK_StimulationId_ExperimentStop, t, 0)
end