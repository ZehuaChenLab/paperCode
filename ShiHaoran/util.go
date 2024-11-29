/**
  @author: 黄睿楠
  @since: 2022/4/21
  @desc: 自定义工具
**/

package main

//Go自带源代码库有两个rand包，同时使用会造成冲突，导入时利用包的别名机制解决此问题
import (
	crypto_rand "crypto/rand"
	"fmt"
	"log"
	"math/big"
	"strconv"
	"strings"
	"time"
)

// 返回一个十位数的随机数，作为消息id
func getRandom() int {
	x := big.NewInt(10000000000)
	for {
		result, err := crypto_rand.Int(crypto_rand.Reader, x)
		if err != nil {
			log.Panic(err)
		}
		if result.Int64() > 1000000000 {
			return int(result.Int64())
		}
	}
}

// // 产生随机值
// func randRange(min, max int64) int64 {
// 	//用于心跳信号的时间
// 	math_rand.Seed(time.Now().UnixNano())
// 	return math_rand.Int63n(max-min) + min
// }

// 获取当前时间的毫秒数
func millisecond() int64 {
	return time.Now().UnixNano() / int64(time.Millisecond)
}

// 提取端口号并且返回uint32
func extractPortNumber(portStr string) (uint32, error) {
	// 检查输入字符串是否为空或不符合预期的格式
	if portStr == "" || !strings.HasPrefix(portStr, ":") {
		return 0, fmt.Errorf("invalid port string format: %s", portStr)
	}

	// 去掉字符串中的":"前缀
	portStr = portStr[1:]

	// 将剩余的字符串转换为uint64
	port, err := strconv.ParseUint(portStr, 10, 32) // 10 表示十进制，32 表示结果需要在 32 位内
	if err != nil {
		return 0, fmt.Errorf("failed to convert port string to integer: %s", portStr)
	}

	// 将结果转换为 uint32
	return uint32(port), nil
}

// 通过端口号查找 nodeTable 中的键
func findNodeKeyByPort(nodeTable map[string]string, port string) (string, bool) {
	for key, value := range nodeTable {
		if value == ":"+port {
			return key, true
		}
	}
	return "", false
}
