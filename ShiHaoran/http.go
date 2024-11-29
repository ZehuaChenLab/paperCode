/**
  @author: 黄睿楠
  @since: 2022/4/21
  @desc: API接口
**/

package main

import (
	"fmt"
	"github.com/gin-gonic/gin"
	"log"
	"net/http"
	"net/rpc"
)

// 8080端口等待客户端访问，若收到请求，则将消息转发给Leader节点
func (rf *Raft) httpListen() {

	// 使用默认中间件（logger 和 recovery 中间件）创建 gin 路由
	r := gin.Default()

	//http://localhost:8080/req?message=xxx
	// /req请求路径 映射到匿名函数
	r.GET("/req",func(c *gin.Context){

		c.JSON(http.StatusOK, gin.H{
			"Status":  "ok",
		})

		if len(c.Query("message")) > 0 && rf.currentLeader != "-1" {

			// 消息封装
			message:=c.Query("message")
			m := new(Message)
			m.MsgID = getRandom()
			m.Msg = message

			// 接收到消息后，转发到领导者
			fmt.Println("http监听到了消息，准备发送给领导者，消息id:", m.MsgID)
			// Leader节点的端口号
			port := nodeTable[rf.currentLeader]

			//客户端用rpc.DialHTTP和RPC服务器进行一个链接(协议必须匹配)
			client, err := rpc.DialHTTP("tcp", "127.0.0.1"+port)
			if err != nil {
				log.Panic(err)
			}
			b := false

			//通过client对象进行远程函数调用
			err = client.Call("Raft.BroadcastMessage", m, &b)
			if err != nil {
				log.Panic(err)
			}
			fmt.Println("消息是否转发到领导者：", b)
		}
	})
	// 监听localhost:8080
	r.Run()
	fmt.Println("监听8080")
}

