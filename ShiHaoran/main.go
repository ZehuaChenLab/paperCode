package main

import (
	"fmt"
	"os"
	"time"
)

func main() {

	//定义三个节点  节点编号 - 端口号
	nodeTable = map[string]string{
		"A": ":9000",
		"B": ":9001",
		"C": ":9002",
		// "D": ":9003",
		// "E": ":9004",
		// "F": ":9005",
		// "G": ":9006",
		// "H": ":9007",
		// "I": ":9008",
	}
	// existingRecords, err := readExistingRecords("stats.csv")
	// if err != nil {
	// 	// 处理错误，例如记录日志或退出程序
	// 	panic(err) // 这里使用 panic 仅为示例，实际应用中应使用更合适的错误处理方式
	// }
	// fmt.Printf("existingRecords: %v\n", existingRecords)
	// rank, maxpr, maxid, _ := processRecords(existingRecords)
	// fmt.Printf("rank: %v\n", rank)
	// fmt.Printf("maxpr: %v\n", maxpr)
	// fmt.Printf("maxid: %v\n", maxid)

	// A B C
	id := os.Args[1]
	fmt.Printf("id: %v\n", id)
	// 创建raft节点实例
	fmt.Printf("当前时间为:%d\n\n", millisecond())
	raft := NewRaft(id, nodeTable[id])

	// rpc服务注册
	go rpcRegister(raft)
	//发送心跳,只有当前节点为Leader节点时，才会开启心跳通道,向其他节点发送心跳
	go raft.heartbeat()
	//开启Http监听，这里设置A节点监听来自8080端口的请求
	if id == "A" {
		go raft.httpListen()
	}

	// 进行第一次选举
	go raft.startElection()

	//进行超时选举
	for {
		// 0.5秒检测一次
		time.Sleep(time.Millisecond * 5000)

		if raft.lastHeartBeatTime != 0 && (millisecond()-raft.lastHeartBeatTime) > int64(heartBeatTimeout*1000) {
			fmt.Printf("心跳检测超时")
			fmt.Println("即将重新开启选举,写入csv文件")
			writeToCSV("stats.csv", communicationRecords)
			raft.reDefault()
			raft.setCurrentLeader("-1")
			raft.lastHeartBeatTime = 0
			go raft.startElection()
		}
	}
}
