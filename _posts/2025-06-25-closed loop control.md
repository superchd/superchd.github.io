---
layout: single
title: "Error/github_io"
categories : ERROR
tag: [github_io_manual, blog, ruby]
toc: true
author_profile : false
sidebar:
     nav: "docs/ERROR"
---

### 루비 버전 충돌로 인한 에러 고치기
> 현재 나는 루비버전을 2개 다운 받아서 터미널을 켤때마다 버전이 충돌하는것 같다.   
> * rbenv global 3.1.2 커맨드를 이용해 버전을 변경해준다  

```
 hyundae  ~/Desktop/superchd.github.io   master ● ?  rbenv global 3.1.2                                       ✔
```  

> * 그 후에, bundle add jekyll을 이용한다.  
> * eval "$(rbenv init -)"  이 커맨드도 필요한것 같다. 다만, 블로그에 올라가면서 큰 따옴표가 바뀌니
> 조심하고 추가 ㅎ
 
 ```
 hyundae  ~/Desktop/superchd.github.io   master ● ?  bundle add jekyll                                        ✔

[!] There was an error parsing `injected gems`: You cannot specify the same gem twice with different version requirements.
You specified: jekyll (~> 4.2) and jekyll (>= 0). Gem already added. Bundler cannot continue.

 #  from injected gems:1
 #  -------------------------------------------
 >  gem "jekyll", ">= 0"
 #  -------------------------------------------
 hyundae  ~/Desktop/superchd.github.io   master ● ?  bundle exec jekyll -v                                  4 ↵
jekyll 4.2.2
 hyundae  ~/Desktop/superchd.github.io   master ● ?  bundle exec jekyll serve --watch                         ✔
Configuration file: /Users/hyundae/Desktop/superchd.github.io/_config.yml
To use retry middleware with Faraday v2.0+, install `faraday-retry` gem
            Source: /Users/hyundae/Desktop/superchd.github.io
       Destination: /Users/hyundae/Desktop/superchd.github.io/_site
 Incremental build: disabled. Enable with --incremental
      Generating... 
       Jekyll Feed: Generating feed for posts
                    done in 0.673 seconds.
 Auto-regeneration: enabled for '/Users/hyundae/Desktop/superchd.github.io'
    Server address: http://127.0.0.1:4000
  Server running... press ctrl-c to stop.
      Regenerating: 1 file(s) changed at 2022-07-22 17:01:54
                    _pages/category-archive.md
       Jekyll Feed: Generating feed for posts
                    ...done in 0.569365 seconds.
```
### 연결이 완료된다. 

                          
