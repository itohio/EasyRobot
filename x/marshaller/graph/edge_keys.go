package graph

import "fmt"

func edgeKey(parentID, childID int64) string {
	return fmt.Sprintf("%d:%d", parentID, childID)
}
