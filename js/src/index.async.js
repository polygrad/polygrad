'use strict'

const main = require('./index')

const api = {}
Object.defineProperties(api, Object.getOwnPropertyDescriptors(main))
api.create = main.createAsync

module.exports = api
