var data = []
var token = ""

fr = new FileReader()
jQuery(document).ready(function () {
    let on_reader_load = (fl) => {
        console.info('. file reader load', fl);
        return display_file; // a function
    };

    $(document.getElementById("uploader-file")).on("change", function () {
        value = this.files[0]
        fr.readAsText(value)

        // fr.onload = () => {
        //     let val = fr.result


        // req = $.ajax({
        //     url: '/predict',
        //     type: 'POST',
        //     dataType: "json",
        //     contentType: "application/json",
        //     data:  JSON.stringify({"sentence_from_file": val, "file": true}),
        //     beforeSend: function () {
        //     $('.overlay').show()
        //     },
        //     complete: function () {
        //         $('.overlay').hide()
        //     }
        //
        //     }).done(function (jsondata, textStatus, jqXHR) {
        //
        //     $('#final-score').val(jsondata['response']['mean_score'])
        //     $('#txt_review').val(jsondata['response']['scores'])
        //     }).fail(function (jsondata, textStatus, jqXHR) {
        //         alert(jsondata['responseJSON'])
        //     });
        //     }
    })

    $('#btn-process').on('click', function () {
        // review = $('#txt_review').val()

        // fr.onload = () => {
        let val = fr.result
        alert(val)
        $.ajax({
            url: '/predict',
            type: "post",
            contentType: "application/json",
            dataType: "json",

            data: JSON.stringify({
                "sentence_from_file": val,
            }),
            beforeSend: function () {
                $('.overlay').show()
            },
            complete: function () {
                $('.overlay').hide()
            }
        }).done(function (jsondata, textStatus, jqXHR) {

            document.getElementById("mean-score").innerText = jsondata['response']['mean_score']
            document.getElementById("mean-is-wordfrom-rep-in-sent-pairs").innerText = jsondata['response']['mean_is_wordfrom_rep_in_sent_pairs']
            document.getElementById("mean-is-deriv-in-sent-pairs").innerText = jsondata['response']['mean_is_deriv_in_sent_pairs']
            document.getElementById("mean-is-hyponym-in-sent-pairs").innerText = jsondata['response']['mean_is_hyponym_in_sent_pairs']
            document.getElementById("mean-is-hypernym-in-sent_pairs").innerText = jsondata['response']['mean_is_hypernym_in_sent_pairs']
            document.getElementById("mean-is-anph-cand-in-sent-pairs").innerText = jsondata['response']['mean_is_anph_cand_in_sent_pairs']


            $('#txt_review').val(val)
        }).fail(function (jsondata, textStatus, jqXHR) {
            alert(jsondata['responseJSON'])
        });

    })




    $('#txt_review').keypress(function (e) {
        if (e.which === 13) {
            $('#btn-process').click()
            e.preventDefault()
        }
    });
})

function toRoutePredictDownloadCsv() {
    window.location = '/download_csv';
}

// $.ajax({
// 	type : "POST",
// 	url : '/predict',
// 	dataType: "json",
// 	// data: JSON.stringify(you can put in a variable in here to send data with the request),
// 	contentType: 'application/json',
// 	success: function () {
// 		alert("")
// 		}
// 	});