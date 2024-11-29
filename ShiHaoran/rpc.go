/**
  @author: 黄睿楠
  @since: 2022/4/21
  @desc: RPC
**/

package main

import (
	"fmt"
	"log"
	"net/http"
	"net/rpc"
	"time"
)

// rpc服务注册
func rpcRegister(raft *Raft) {
	//注册一个raft节点
	err := rpc.Register(raft)
	if err != nil {
		log.Panic(err)
	}
	port := raft.node.Port
	//把服务绑定到http协议上
	rpc.HandleHTTP()
	//监听端口
	err = http.ListenAndServe(port, nil)
	if err != nil {
		fmt.Println("注册rpc服务失败", err)
	}
}

// TODO   第三个参数是一个匿名函数
func (rf *Raft) broadcast(method string, args interface{}, fun func(ok bool)) {

	// 遍历除自己以外的所有node
	for nodeID, port := range nodeTable {
		if nodeID == rf.node.ID {
			continue
		}
		//连接远程rpc
		rp, err := rpc.DialHTTP("tcp", "127.0.0.1"+port)
		if err != nil {
			fun(false)
			continue
		}

		var bo = false
		err = rp.Call(method, args, &bo)
		if err != nil {
			fun(false)
			continue
		}
		fun(bo)
	}
}

// 以下三个方法在raft.go中，通过broadcast方法调用
// 投票
func (rf *Raft) Vote(node NodeInfo, b *bool) error {
	if rf.votedFor != "-1" || rf.currentLeader != "-1" {
		*b = false
	} else {

		rf.setVoteFor(node.ID)
		fmt.Printf("投票成功，已投%s节点\n", node.ID)
		recordCommunication(rf.node.Port, node.Port, 1)
		fmt.Printf("投票成功更新communicationRecords: %v\n", communicationRecords)
		*b = true
	}
	return nil
}

// 第一次收到Leader的心跳：
func (rf *Raft) ReceiveFirstHeartbeat(node NodeInfo, b *bool) error {
	rf.setCurrentLeader(node.ID)
	rf.reDefault()
	fmt.Println("已发现网络中的领导节点，", node.ID, "成为了领导者！")
	*b = true
	return nil
}

// 之后收到Leader的心跳：
func (rf *Raft) ReceiveHeartbeat(node NodeInfo, b *bool) error {
	rf.setCurrentLeader(node.ID)
	rf.lastHeartBeatTime = millisecond()
	fmt.Printf("接收到来自领导节点%s的心跳检测\n", node.ID)
	recordCommunication(nodeTable[rf.currentLeader], rf.node.Port, 1)
	writeToCSV("stats.csv", communicationRecords)
	fmt.Printf("当前时间为:%d\n\n", millisecond())

	*b = true
	return nil
}

// Leader节点的日志复制
// http.go调用，领导者收到了追随者节点转发过来的消息
func (rf *Raft) BroadcastMessage(message Message, b *bool) error {
	fmt.Printf("领导者节点接收到客户端的消息，id为:%d\n", message.MsgID)
	rf.MessageStore[message.MsgID] = message.Msg
	*b = true
	fmt.Println("准备将消息进行广播...")
	num := 0
	go rf.broadcast("Raft.ReceiveMessage", message, func(ok bool) {
		if ok {
			num++

		}
	})

	for {
		//自己默认收到了消息，所以减1
		if num > raftCount/2-1 {
			fmt.Printf("全网已超过半数节点接收到消息id：%d\nraft验证通过，可以打印消息\n", message.MsgID)
			fmt.Println("消息为：", rf.MessageStore[message.MsgID])
			rf.lastMessageTime = millisecond()
			fmt.Println("通知客户端：消息提交成功")
			go rf.broadcast("Raft.ConfirmationMessage", message, func(ok bool) {
			})
			break
		} else {
			//休息会儿
			time.Sleep(time.Millisecond * 100)
		}
	}
	return nil
}

// rpc.go BroadcastMessage方法调用，追随者接收消息，然后存储到数据库中，待领导者确认后打印
func (rf *Raft) ReceiveMessage(message Message, b *bool) error {
	fmt.Printf("接收到领导者节点发来的信息，消息id为：%d\n", message.MsgID)
	rf.MessageStore[message.MsgID] = message.Msg
	*b = true
	fmt.Println("已回复接收到消息，待领导者确认后打印")
	recordCommunication(nodeTable[rf.currentLeader], rf.node.Port, 1)
	fmt.Printf("日志复制更新communicationRecords: %v\n", communicationRecords)
	return nil
}

// rpc.go BroadcastMessage调用
// 追随者节点的反馈得到领导者节点的确认，开始打印消息
func (rf *Raft) ConfirmationMessage(message Message, b *bool) error {
	for {
		if _, ok := rf.MessageStore[message.MsgID]; ok {
			fmt.Printf("raft验证通过，可以打印消息，消息id为：%d\n", message.MsgID)
			fmt.Println("消息为：", rf.MessageStore[message.MsgID])
			rf.lastMessageTime = millisecond()
			break
		} else {
			//如果没有此消息，等一会再看看
			time.Sleep(time.Millisecond * 10)
		}

	}
	*b = true
	return nil
}
