package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
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
	flag.Parse()

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

	t := template.Must(template.New("code").Parse(string(tplContent)))

	for _, entry := range cfg {
		err := generate(t, entry)
		if err != nil {
			fmt.Println(entry.FileName, " Failed with: ", err)
		} else {
			fmt.Println(entry.FileName, " OK")
		}

		if entry.PostProcess != nil {
			cmd := exec.Command(entry.PostProcess[0], entry.PostProcess[1:]...)
			err := cmd.Run()
			if err != nil {
				fmt.Println(entry.FileName, " post process failed: ", err)
			}
		}
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
