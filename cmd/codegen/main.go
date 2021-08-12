package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"text/template"
)

type Config struct {
	FileName    string                 `json:"file_name"`
	ExtraCode   string                 `json:"extra_code"`
	PostProcess []string               `json:"post_process"`
	Params      map[string]interface{} `json:"params"`
}

func main() {
	tpl := flag.String("i", "", "Path to template file")
	config := flag.String("c", "", "Path to configuration file")
	mode := flag.String("m", "template", "Generation mode: one of (template, replace)")
	flag.Parse()

	if *tpl == "" || *config == "" {
		flag.PrintDefaults()
		return
	}

	fTpl, err := os.Open(*tpl)
	if err != nil {
		panic(err)
	}
	defer fTpl.Close()
	fCfg, err := os.Open(*config)
	if err != nil {
		panic(err)
	}
	defer fCfg.Close()

	tplContent, err := io.ReadAll(fTpl)
	if err != nil {
		panic(err)
	}
	cfgContent, err := io.ReadAll(fCfg)
	if err != nil {
		panic(err)
	}

	var cfg []Config
	json.Unmarshal(cfgContent, &cfg)

	switch {
	case strings.HasPrefix("template", *mode):
		generateTemplate(tplContent, cfg)
	case strings.HasPrefix("replace", *mode):
		generateReplace(tplContent, cfg)
	default:
		fmt.Println("Valid mode must be specified")
	}
}

func generateTemplate(tplContent []byte, cfg []Config) {
	t := template.Must(template.New("code").Parse(string(tplContent)))

	for _, entry := range cfg {
		err := generate(t, entry)
		if err != nil {
			fmt.Println(entry.FileName, " Failed with: ", err)
		} else {
			fmt.Println(entry.FileName, " OK")
		}

		postProcess(entry)
	}
}

func generateReplace(tplContent []byte, cfg []Config) {
	for _, entry := range cfg {
		err := replace(tplContent, entry)
		if err != nil {
			fmt.Println(entry.FileName, " Failed with: ", err)
		} else {
			fmt.Println(entry.FileName, " OK")
		}

		postProcess(entry)
	}
}

func generate(t *template.Template, c Config) error {
	f, err := os.Create(c.FileName)
	if err != nil {
		return err
	}
	defer f.Close()
	return t.Execute(f, c.Params)
}

func joinStrings(a interface{}, regex bool) (string, error) {
	if s, ok := a.(string); ok {
		return s, nil
	}

	arr, ok := a.([]interface{})
	if !ok {
		return "", fmt.Errorf("Not a list of strings")
	}

	strArr := make([]string, 0, len(arr))
	for _, s := range arr {
		if str, ok := s.(string); ok {
			strArr = append(strArr, str)
		}
	}

	if len(strArr) == 0 {
		return "", fmt.Errorf("Not a list of strings")
	}

	if regex {
		return strings.Join(strArr, ""), nil
	}
	return strings.Join(strArr, "\n"), nil
}

func replace(content []byte, c Config) error {
	str := string(content)

	f, err := os.Create(c.FileName)
	if err != nil {
		return err
	}
	defer f.Close()

	for orig, new := range c.Params {
		if newS, ok := new.(string); ok {
			str = strings.ReplaceAll(str, orig, newS)
		} else if new, ok := new.(map[string]interface{}); ok {
			old, ok := new["old"]
			if !ok {
				continue
			}
			rxtype, regex := new["regex"]
			new, ok := new["new"]
			if !ok {
				continue
			}

			oldS, err := joinStrings(old, regex)
			if err != nil {
				continue
			}
			newSS, err := joinStrings(new, false)
			if err != nil {
				continue
			}

			if regex {
				str = applyRegex(str, oldS, newSS, rxtype)
			} else {
				fmt.Println(oldS, " -> ", newSS)
				str = strings.ReplaceAll(str, oldS, newSS)
			}
		}
	}

	_, err = f.Write([]byte(str))
	return err
}

func applyRegex(str, old, new string, rxtype interface{}) string {
	if what, ok := rxtype.(string); ok && what == "func" {
		re := regexp.MustCompile(fmt.Sprintf("func.*\\s%s\\(.*{", old))
		loc := re.FindStringIndex(str)
		if loc == nil {
			return str
		}
		start := loc[0]
		brackets := 1
		for i := loc[1]; i < len(str); i++ {
			if str[i] == '{' {
				brackets++
			} else if str[i] == '}' {
				brackets--
				if brackets == 0 {
					return strings.ReplaceAll(str, str[start:i+1], new)
				}
			}
		}
	}

	re := regexp.MustCompile(old)
	return re.ReplaceAllString(str, new)
}

func postProcess(entry Config) {
	if entry.PostProcess == nil {
		return
	}

	cmd := exec.Command(entry.PostProcess[0], entry.PostProcess[1:]...)
	err := cmd.Run()
	if err != nil {
		fmt.Println(entry.FileName, " post process failed: ", err)
	}
}
