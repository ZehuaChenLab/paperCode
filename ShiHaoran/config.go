package main

// 节点数量
var raftCount = 3

// 节点集   key:节点编号  value:端口号
var nodeTable map[string]string

// 选举超时时间，如果一个跟随者在一段时间里没有接收到任何消息，也就是选举超时
// 那么他就会认为系统中没有可用的领导人,并且发起选举以选出新的领导人
var heartBeatTimeout = 7

// 心跳检测频率（单位：秒）
var heartBeatTimes = 3

// 一个term的时间（单位：秒），从一个跟随者变成候选人开始计时，经过term时间后，如果没有成为领导者，则结束选举过程，将自己变回跟随者
var timeout = 3

// 初始化通信记录映射
var communicationRecords = make(map[uint32]map[uint32]float64)

// var t = 3         // 阈值
// var totalKeys = 5 // 总共的私钥数量
