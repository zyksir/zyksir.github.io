# [Zhu Yikai Blog](https://zyksir.github.io)

================================

## Getting Started

1. [install ruby](https://ruby-lang.org/en/documentation/installation). For MacOS users, you should follow this [guide](https://mac.install.guide/ruby/13.html). After this step, if you run `ruby --version`, you will see a version greater than 3.
2. fork this github and `git clone` into you local computer. note the name should be `[YOUR-GITHUB-USERNAME].github.io`!
3. run `bundle install` inside the github directory.
4. check your webite locally. run `bundle exec jekyll serve`
5. By now, you should see your website with "http://127.0.0.1:4000/"

## Replace my info with yours

1. in [_config.yml](./_config.yml), change every personal information to yours.
2. to enable disqus comment, I go to https://disqus.com/, click `get started`, click `I want to install disqus on my website`. follow their instructions you will get a js code, just copy and paste by searching.`disqus_enable`. you can watch [this video](https://disqus.com/admin/install/platforms/universalcode/). They have free services and it's enough for my use case.
3. to enbale utterances comment, go to [utterances](https://github.com/apps/utterances) and install it. After follow this website you will get a js code, just copy and paste by searching `utterances_enable`.
中文版见我的第一个 post(for chinese version, see my first post).

## Post New Blog

```bash
rake post title="TITLE" subtitle="SUBTITLE"
```