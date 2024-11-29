package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/alixaxel/pagerank"
)

// 记录节点间通信的函数
func recordCommunication(sourcePort, targetPort string, count int) {
	// 提取端口号
	sourcePortUint32, err := extractPortNumber(sourcePort)
	if err != nil {
		// 处理错误，例如记录日志
		log.Printf("Error extracting source port: %v", err)
		return
	}
	targetPortUint32, err := extractPortNumber(targetPort)
	if err != nil {
		// 处理错误，例如记录日志
		log.Printf("Error extracting target port: %v", err)
		return
	}

	// 检查源节点是否已存在记录，如果不存在则初始化
	if _, exists := communicationRecords[sourcePortUint32]; !exists {
		communicationRecords[sourcePortUint32] = make(map[uint32]float64)
	}

	// 增加或更新源节点与目标节点之间的通信次数
	communicationRecords[sourcePortUint32][targetPortUint32] += float64(count)
}

// 写入通信记录到 CSV 文件的函数
func writeToCSV(filename string, records map[uint32]map[uint32]float64) error {
	// 打开文件，如果文件不存在则创建
	file, err := os.OpenFile(filename, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0644)
	if err != nil {
		return err
	}
	defer file.Close()

	// 创建 CSV 写入器
	writer := csv.NewWriter(file)
	defer writer.Flush()

	// 遍历映射并写入 CSV 文件
	for sourceNodeID, targetRecords := range records {
		for targetNodeID, count := range targetRecords {
			// 准备写入的数据切片
			data := []string{
				// time.Now().Format(time.RFC3339), // 记录时间
				strconv.FormatUint(uint64(sourceNodeID), 10),
				strconv.FormatUint(uint64(targetNodeID), 10),
				strconv.FormatFloat(float64(count), 'f', -1, 64),
			}

			// 写入一行数据到 CSV 文件
			if err := writer.Write(data); err != nil {
				return err
			}
		}
	}

	return nil
}

// 用于从 CSV 文件中读取现有的通信记录
func readExistingRecords(filename string) (map[uint32]map[uint32]float64, error) {
	records := make(map[uint32]map[uint32]float64)
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file %s: %v", filename, err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	recordsRaw, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("failed to read all records from file %s: %v", filename, err)
	}

	for _, record := range recordsRaw {
		if len(record) != 3 {
			fmt.Printf("Skipping invalid record: %v\n", record)
			continue // 跳过格式不正确的记录
		}
		sourceNodeID, err := strconv.ParseUint(record[0], 10, 32)
		if err != nil {
			fmt.Printf("Skipping record with invalid source node ID: %s\n", record[0])
			continue // 跳过无法解析的源节点 ID
		}
		targetNodeID, err := strconv.ParseUint(record[1], 10, 32)
		if err != nil {
			fmt.Printf("Skipping record with invalid target node ID: %s\n", record[1])
			continue // 跳过无法解析的目标节点 ID
		}
		count, err := strconv.ParseFloat(record[2], 64)
		if err != nil {
			fmt.Printf("Skipping record with invalid count: %s\n", record[2])
			continue // 跳过无法解析的通信次数
		}
		if _, exists := records[uint32(sourceNodeID)]; !exists {
			records[uint32(sourceNodeID)] = make(map[uint32]float64)
		}
		records[uint32(sourceNodeID)][uint32(targetNodeID)] += count
	}

	return records, nil
}

// processRecords 处理 existingRecords，构建 PageRank 图，并输出所有节点的 PageRank 值
// 同时找到并返回具有最大 PageRank 值的节点的端口号
func processRecords(existingRecords map[uint32]map[uint32]float64) (map[uint32]float64, uint32, string, error) {
	// 创建 PageRank 图
	graph := pagerank.NewGraph()

	// 提取 records 中的源端口号、目标端口号和通信次数
	for srcPort, tgtPorts := range existingRecords {
		for tgtPort, count := range tgtPorts {
			// 将端口号转换为 uint32 并添加边到图中
			graph.Link(uint32(srcPort), uint32(tgtPort), float64(count))
		}
	}

	// 计算 PageRank 值
	ranks := make(map[uint32]float64)
	maxRank := float64(0)
	maxPRNode := uint32(0)
	graph.Rank(0.85, 0.000001, func(node uint32, rank float64) {
		ranks[node] = rank
		// 检查并更新最大 PageRank 值和对应的节点
		if rank > maxRank {
			maxRank = rank
			maxPRNode = node
		}
	})

	maxprStr := strconv.Itoa(int(maxPRNode))
	maxid, _ := findNodeKeyByPort(nodeTable, maxprStr)

	return ranks, maxPRNode, maxid, nil
}
