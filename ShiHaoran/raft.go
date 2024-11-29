package main

import (
	"fmt"
	"sync"
	"time"
)

// raft节点的属性
type Raft struct {

	// 节点的id和端口号
	node *NodeInfo

	//本节点获得的投票数
	vote int

	//互斥锁
	lock sync.Mutex

	//当前任期
	currentTerm int

	//为哪个节点投票
	votedFor string

	//当前节点的领导
	currentLeader string

	//当前节点状态   0 follower  1 candidate  2 leader
	state int

	//发送最后一条消息的时间
	lastMessageTime int64

	//最后一次心跳检测时间
	lastHeartBeatTime int64

	//接收投票成功通道
	voteCh chan bool

	//心跳信号
	heartBeat chan bool

	//该节点的数据库，每个节点都有自己的数据库
	MessageStore map[int]string

	Maxid string

	// //私钥
	// privateKey bls.SecretKey
	// //公钥
	// publicKey bls.PublicKey
}

type NodeInfo struct {
	ID   string
	Port string
}

type Message struct {
	Msg   string
	MsgID int
}

// 构造器
func NewRaft(id, port string) *Raft {

	node := new(NodeInfo)
	node.ID = id
	node.Port = port

	rf := new(Raft)

	//节点信息
	rf.node = node

	//当前节点获得票数
	rf.setVote(0)

	//设置任期
	rf.setTerm(0)

	//给0  1  2三个节点投票，给谁都不投
	rf.setVoteFor("-1")

	//最初没有领导
	rf.setCurrentLeader("-1")

	//0 follower
	rf.setStatus(0)

	//最后一次心跳检测时间
	rf.lastHeartBeatTime = 0

	//定义两个阻塞通道
	//投票通道
	rf.voteCh = make(chan bool)

	//心跳通道
	rf.heartBeat = make(chan bool)

	// rf.privateKey = generatePrivateKeys(1)[0]
	// rf.publicKey = generatePublicKeys(generatePrivateKeys(1))[0]

	// 初始化数据库
	rf.MessageStore = make(map[int]string)

	existingRecords, err := readExistingRecords("stats.csv")
	if err != nil {
		// 处理错误，例如记录日志或退出程序
		panic(err) // 这里使用 panic 仅为示例，实际应用中应使用更合适的错误处理方式
	}

	fmt.Printf("existingRecords: %v\n", existingRecords)

	rank, maxpr, maxid, _ := processRecords(existingRecords)
	fmt.Printf("rank: %v\n", rank)
	fmt.Printf("maxpr: %v\n", maxpr)
	fmt.Printf("maxid: %v\n", maxid)

	rf.Maxid = maxid
	fmt.Printf("当前时间为:%d\n\n", millisecond())
	return rf
}

// 完整的选举方法:分为两步，先切换为候选人，然后进行选举操作
func (rf *Raft) startElection() {
	fmt.Printf("开始选举当前时间为:%d\n\n", millisecond())
	for {
		// 将节点状态从 跟随者 切换为 候选人
		if rf.becomeCandidate() {
			//成为候选人节点后 向其他节点要票，如果raft节点顺利成为Leader，则退出for循环，结束选举
			if rf.becomeLeader() {
				break
			} else {
				// 因为选举超时没有成为Leader，可能选票被瓜分，也可能别人抢先成为Leader
				// 所以不确定现在全网有没有Leader，不能直接break，而应该继续尝试变成candidate
				continue
			}
		} else { // 如果raft节点不是候选人，则退出for循环
			break
		}
	}
}

// 修改节点为候选人状态，并给自己投票,Term+1
func (rf *Raft) becomeCandidate() bool {

	// config中定义的选举超时时间是固定的，为了防止选票被不断瓜分，这里将选举超时时间进行随机化处理
	// 生成在[1500,5000)范围内的随机时间
	// r := randRange(1500, 5000)
	// //经过随机时间后，开始成为候选人
	// time.Sleep(time.Duration(r) * time.Millisecond)
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

	//如果发现本节点已经投过票，或者已经存在领导者，则返回false
	if rf.state == 0 && rf.currentLeader == "-1" && rf.votedFor == "-1" && rf.Maxid == rf.node.ID {

		//将节点状态变为1
		rf.setStatus(1)
		//给自己投票
		rf.setVoteFor(rf.node.ID)
		rf.voteAdd()
		//节点任期加1
		rf.setTerm(rf.currentTerm + 1)
		//当前没有领导
		rf.setCurrentLeader("-1")

		fmt.Println("本节点已变更为候选人状态")
		fmt.Printf("当前时间为:%d\n\n", millisecond())
		return true
	} else {
		return false
	}
}

// 进行选举操作，若当前节点在选举超时之前获得超过半数的选票，则返回true，若选举超时，返回false，只有这两种情况
func (rf *Raft) becomeLeader() bool {
	fmt.Println("开始选举领导，向其他节点进行广播")
	fmt.Printf("当前时间为:%d\n\n", millisecond())
	go rf.broadcast("Raft.Vote", rf.node, func(ok bool) {
		rf.voteCh <- ok
	})
	for {
		// select多路复用:
		// select中如果没有default语句，则会阻塞等待任一case
		// select语句中除default外，各case的执行顺序是完全随机的
		select {
		// func After(d Duration) <-chan Time:等待d时间之后，将当前时间发送到通道中
		case <-time.After(time.Second * time.Duration(timeout)):
			fmt.Println("领导者选举超时，节点变更为追随者状态")
			rf.reDefault()
			return false

		case ok := <-rf.voteCh:
			if ok {
				rf.voteAdd()
				fmt.Printf("获得来自其他节点的投票，当前得票数：%d\n", rf.vote)
			}
			if rf.vote > raftCount/2 && rf.currentLeader == "-1" {
				fmt.Println("获得超过网络节点二分之一的得票数，本节点被选举成为了leader")
				fmt.Printf("当前时间为:%d\n\n", millisecond())
				//节点状态变为leader
				rf.setStatus(2)
				rf.setCurrentLeader(rf.node.ID)
				fmt.Println("向其他节点进行广播...")
				//使其他节点更新领导者信息，并切换为跟随者
				go rf.broadcast("Raft.ReceiveFirstHeartbeat", rf.node, func(ok bool) {
					fmt.Println(ok)
				})
				//开启心跳检测通道
				rf.heartBeat <- true
				return true
			}
			// 若该节点目前没有被选举为Leader，比如票数不够，或其他节点抢先成为Leader，则阻塞
			// 接下来有两种情况：要么等到票数够了，自己成为Leader，返回true，要么别人先成为了Leader，自己阻塞到超时，返回false
		}
	}
}

// 发送心跳
func (rf *Raft) heartbeat() {
	//如果收到通道开启的信息,将会向其他节点进行固定频率的心跳发送
	if <-rf.heartBeat {
		// 无限循环，间隔发送心跳
		for {
			fmt.Println("本节点开始发送心跳检测...")
			rf.broadcast("Raft.ReceiveHeartbeat", rf.node, func(ok bool) {
				fmt.Println("收到回复:", ok)

			})
			rf.lastHeartBeatTime = millisecond()
			// 心跳检测频率
			time.Sleep(time.Second * time.Duration(heartBeatTimes))
		}
	}
}

// 恢复默认设置
func (rf *Raft) reDefault() {
	rf.setVote(0)
	rf.setVoteFor("-1")
	rf.setStatus(0)
	//rf.currentLeader = "-1"
}

//============= set方法，原子操作 =============

// 设置任期
func (rf *Raft) setTerm(term int) {
	rf.lock.Lock()
	rf.currentTerm = term
	rf.lock.Unlock()
}

// 设置为谁投票
func (rf *Raft) setVoteFor(id string) {
	rf.lock.Lock()
	rf.votedFor = id
	rf.lock.Unlock()
}

// 设置当前领导者
func (rf *Raft) setCurrentLeader(leader string) {
	rf.lock.Lock()
	rf.currentLeader = leader
	rf.lock.Unlock()
}

// 设置当前状态
func (rf *Raft) setStatus(state int) {
	rf.lock.Lock()
	rf.state = state
	rf.lock.Unlock()
}

// 投票累加
func (rf *Raft) voteAdd() {
	rf.lock.Lock()
	rf.vote++
	rf.lock.Unlock()
}

// 设置投票数量
func (rf *Raft) setVote(num int) {
	rf.lock.Lock()
	rf.vote = num
	rf.lock.Unlock()
}
